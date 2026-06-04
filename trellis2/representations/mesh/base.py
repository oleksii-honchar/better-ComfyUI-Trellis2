from typing import *
import torch
from ..voxel import Voxel
import cumesh
from flex_gemm.ops.grid_sample import grid_sample_3d

import numpy as np
import meshlib.mrmeshnumpy as mrmeshnumpy
import meshlib.mrmeshpy as mrmeshpy

import gc


class Mesh:
    def __init__(self,
        vertices,
        faces,
        vertex_attrs=None
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.vertex_attrs = vertex_attrs
        
    @property
    def device(self):
        return self.vertices.device
        
    def to(self, device, non_blocking=False):
        return Mesh(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.vertex_attrs.to(device, non_blocking=non_blocking) if self.vertex_attrs is not None else None,
        )
        
    def cuda(self, non_blocking=False):
        return self.to('cuda', non_blocking=non_blocking)
        
    def cpu(self):
        return self.to('cpu')
    
    def fill_holes(self, max_hole_perimeter=3e-2):
        """Fill holes using trimesh (replaces cuMesh — broken on PyTorch 2.11 + Blackwell)."""
        import trimesh
        vertices_np = self.vertices.detach().cpu().numpy().copy()
        faces_np = self.faces.detach().cpu().numpy().copy()
        tm = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
        if tm.is_watertight:
            del tm
            return
        tm.fill_holes()
        tm.remove_unreferenced_vertices()
        tm.update_faces(tm.nondegenerate_faces())
        self.vertices = torch.from_numpy(tm.vertices.copy()).float().to(self.device)
        self.faces = torch.from_numpy(tm.faces.copy()).int().to(self.device)
        del tm
        gc.collect()
        
    def remove_faces(self, face_mask: torch.Tensor):
        """Remove faces using torch boolean indexing (replaces cuMesh)."""
        keep_idx = torch.where(~face_mask)[0] if face_mask.dtype == torch.bool else torch.arange(len(self.faces), device=self.device)
        self.faces = self.faces[keep_idx]
        # Clean up unreferenced vertices
        used_vertices = torch.unique(self.faces.reshape(-1))
        old_to_new = torch.full((len(self.vertices),), -1, dtype=torch.long, device=self.device)
        old_to_new[used_vertices] = torch.arange(len(used_vertices), device=self.device)
        self.faces = old_to_new[self.faces]
        self.vertices = self.vertices[used_vertices]
        
    def simplify_with_cumesh(self, target=1000000, verbose: bool=True, options: dict={}):
        """Simplify using trimesh quadratic decimation (replaces cuMesh — broken on PyTorch 2.11 + Blackwell)."""
        import trimesh
        current_faces_num = len(self.faces)
        if verbose:
            print(f'Current Faces Number: {current_faces_num}')
        if current_faces_num < target:
            return
        
        vertices_np = self.vertices.detach().cpu().numpy().copy()
        faces_np = self.faces.detach().cpu().numpy().copy()
        tm = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
        loss_threshold = options.get('loss_threshold', 1e-4)
        if verbose:
            print(f'Simplifying [thres={loss_threshold:.2e}]...')
        tm_simplified = tm.simplify_quadratic_decimation(int(target))
        if tm_simplified is not None:
            tm = tm_simplified
        tm.remove_unreferenced_vertices()
        tm.update_faces(tm.nondegenerate_faces())
        self.vertices = torch.from_numpy(tm.vertices.copy()).float().to(self.device)
        self.faces = torch.from_numpy(tm.faces.copy()).int().to(self.device)
        del tm; gc.collect()
        
    def simplify_with_meshlib(self, target=1000000):
        current_faces_num = len(self.faces)
        print(f'Current Faces Number: {current_faces_num}')
        
        if current_faces_num<target:
            return

        settings = mrmeshpy.DecimateSettings()
        faces_to_delete = current_faces_num - target
        settings.maxDeletedFaces = faces_to_delete                        
        settings.packMesh = True
        
        print('Generating Meshlib Mesh ...')
        mesh = mrmeshnumpy.meshFromFacesVerts(self.faces.cpu().numpy(), self.vertices.cpu().numpy())
        print('Packing Optimally ...')
        mesh.packOptimally()
        print('Decimating ...')
        mrmeshpy.decimateMesh(mesh, settings)
        
        new_vertices = mrmeshnumpy.getNumpyVerts(mesh)
        new_faces = mrmeshnumpy.getNumpyFaces(mesh.topology)               
        
        print(f"Reduced faces, resulting in {len(new_vertices)} vertices and {len(new_faces)} faces")
        
        self.vertices = torch.from_numpy(new_vertices).float().to(self.device)
        self.faces = torch.from_numpy(new_faces).int().to(self.device)

        del mesh
        gc.collect()


class TextureFilterMode:
    CLOSEST = 0
    LINEAR = 1


class TextureWrapMode:
    CLAMP_TO_EDGE = 0
    REPEAT = 1
    MIRRORED_REPEAT = 2


class AlphaMode:
    OPAQUE = 0
    MASK = 1
    BLEND = 2


class Texture:
    def __init__(
        self,
        image: torch.Tensor,
        filter_mode: TextureFilterMode = TextureFilterMode.LINEAR,
        wrap_mode: TextureWrapMode = TextureWrapMode.REPEAT
    ):
        self.image = image
        self.filter_mode = filter_mode
        self.wrap_mode = wrap_mode

    def to(self, device, non_blocking=False):
        return Texture(
            self.image.to(device, non_blocking=non_blocking),
            self.filter_mode,
            self.wrap_mode,
        )


class PbrMaterial:
    def __init__(
        self,
        base_color_texture: Optional[Texture] = None,
        base_color_factor: Union[torch.Tensor, List[float]] = [1.0, 1.0, 1.0],
        metallic_texture: Optional[Texture] = None,
        metallic_factor: float = 1.0,
        roughness_texture: Optional[Texture] = None,
        roughness_factor: float = 1.0,
        alpha_texture: Optional[Texture] = None,
        alpha_factor: float = 1.0,
        alpha_mode: AlphaMode = AlphaMode.OPAQUE,
        alpha_cutoff: float = 0.5,
    ):
        self.base_color_texture = base_color_texture
        self.base_color_factor = torch.tensor(base_color_factor, dtype=torch.float32)[:3]
        self.metallic_texture = metallic_texture
        self.metallic_factor = metallic_factor
        self.roughness_texture = roughness_texture
        self.roughness_factor = roughness_factor
        self.alpha_texture = alpha_texture
        self.alpha_factor = alpha_factor
        self.alpha_mode = alpha_mode
        self.alpha_cutoff = alpha_cutoff

    def to(self, device, non_blocking=False):
        return PbrMaterial(
            base_color_texture=self.base_color_texture.to(device, non_blocking=non_blocking) if self.base_color_texture is not None else None,
            base_color_factor=self.base_color_factor.to(device, non_blocking=non_blocking),
            metallic_texture=self.metallic_texture.to(device, non_blocking=non_blocking) if self.metallic_texture is not None else None,
            metallic_factor=self.metallic_factor,
            roughness_texture=self.roughness_texture.to(device, non_blocking=non_blocking) if self.roughness_texture is not None else None,
            roughness_factor=self.roughness_factor,
            alpha_texture=self.alpha_texture.to(device, non_blocking=non_blocking) if self.alpha_texture is not None else None,
            alpha_factor=self.alpha_factor,
            alpha_mode=self.alpha_mode,
            alpha_cutoff=self.alpha_cutoff,
        )


class MeshWithPbrMaterial(Mesh):
    def __init__(self,
        vertices,
        faces,
        material_ids,
        uv_coords,
        materials: List[PbrMaterial],
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.material_ids = material_ids    # [M]
        self.uv_coords = uv_coords          # [M, 3, 2]
        self.materials = materials
        self.layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }

    def to(self, device, non_blocking=False):
        return MeshWithPbrMaterial(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.material_ids.to(device, non_blocking=non_blocking),
            self.uv_coords.to(device, non_blocking=non_blocking),
            [material.to(device, non_blocking=non_blocking) for material in self.materials],
        )


class MeshWithVoxel(Mesh, Voxel):
    def __init__(self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        origin: list,
        voxel_size: float,
        coords: torch.Tensor,
        attrs: torch.Tensor,
        voxel_shape: torch.Size,
        layout: Dict = {},
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.origin = torch.tensor(origin, dtype=torch.float32, device=self.device)
        self.voxel_size = voxel_size
        self.coords = coords
        self.attrs = attrs
        self.voxel_shape = voxel_shape
        self.layout = layout

    def to(self, device, non_blocking=False):
        return MeshWithVoxel(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.origin.tolist(),
            self.voxel_size,
            self.coords.to(device, non_blocking=non_blocking),
            self.attrs.to(device, non_blocking=non_blocking),
            self.voxel_shape,
            self.layout,
        )
        
    def query_attrs(self, xyz):
        grid = ((xyz - self.origin) / self.voxel_size).reshape(1, -1, 3)
        vertex_attrs = grid_sample_3d(
            self.attrs,
            torch.cat([torch.zeros_like(self.coords[..., :1]), self.coords], dim=-1),
            self.voxel_shape,
            grid,
            mode='trilinear'
        )[0]
        return vertex_attrs
        
    def query_vertex_attrs(self):
        return self.query_attrs(self.vertices)
