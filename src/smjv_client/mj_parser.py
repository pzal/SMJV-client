import logging
from typing import Dict, List, Tuple

import mujoco
import numpy as np
import trimesh
from mujoco import mj_id2name, mjtObj
from simpub.parser.mj import (
    MJModelGeomTypeMap,
    mj2unity_pos,
    mj2unity_quat,
    scale2unity,
)
from simpub.parser.simdata import (
    SimMesh,
    SimObject,
    SimScene,
    SimSceneConfig,
    SimTransform,
    SimVisual,
    TreeNode,
    VisualType,
    create_material,
    create_texture,
)

logger = logging.getLogger(__name__)


class MjSceneParser:
    def __init__(self, model, visible_geoms_groups, no_rendered_objects=None):
        self.model = model
        self.visible_geoms_groups = visible_geoms_groups
        self.no_rendered_objects = no_rendered_objects or []
        self.sim_scene = self._parse(model)

    def parse(self) -> SimScene:
        return self.sim_scene

    def _parse(self, model) -> SimScene:
        sim_scene = SimScene(
            SimSceneConfig(
                name="MujocoScene", pos=[0, 0, 0], rot=[0, 0, 0, 1], scale=[1, 1, 1]
            )
        )
        hierarchy: Dict[int, Tuple[int, TreeNode]] = {}
        for body_id in range(model.nbody):
            node, parent_id = self._process_body(model, body_id)
            hierarchy[body_id] = (parent_id, node)
        for body_id, (parent_id, node) in hierarchy.items():
            if parent_id == -1:
                sim_scene.root = node
            if parent_id in hierarchy:
                hierarchy[parent_id][1].children.append(node)
        assert sim_scene.root is not None, "SimScene root is None."
        return sim_scene

    def _process_body(self, model, body_id: int) -> Tuple[TreeNode, int]:
        body_name = mj_id2name(model, mjtObj.mjOBJ_BODY, body_id)
        parent_id = int(model.body_parentid[body_id])
        parent_name = "root"
        if parent_id != -1:
            parent_name = mj_id2name(model, mjtObj.mjOBJ_BODY, parent_id)
        if parent_id == body_id:  # world body is its own parent -> root
            parent_id = -1

        trans = SimTransform(
            pos=mj2unity_pos(model.body_pos[body_id].tolist()),
            rot=mj2unity_quat(model.body_quat[body_id].tolist()),
            scale=[1, 1, 1],
        )

        visuals: List[SimVisual] = []
        if (
            int(model.body_geomadr[body_id]) != -1
            and body_name not in self.no_rendered_objects
        ):
            adr = int(model.body_geomadr[body_id])
            for geom_id in range(adr, adr + int(model.body_geomnum[body_id])):
                if int(model.geom_group[geom_id]) not in self.visible_geoms_groups:
                    continue
                visual = self._process_geom(model, geom_id)
                if visual["name"] is None:
                    visual["name"] = f"{body_name}_geom_{geom_id}"
                visuals.append(visual)

        node = TreeNode()
        node.data = SimObject(
            name=body_name, parent=parent_name, trans=trans, visuals=visuals
        )
        return node, parent_id

    def _process_geom(self, model, geom_id: int) -> SimVisual:
        geom_name = mj_id2name(model, mjtObj.mjOBJ_GEOM, geom_id)
        geom_type = int(model.geom_type[geom_id])
        visual_type = MJModelGeomTypeMap.get(geom_type, VisualType.NONE)
        trans = SimTransform(
            pos=mj2unity_pos(model.geom_pos[geom_id].tolist()),
            rot=mj2unity_quat(model.geom_quat[geom_id].tolist()),
            scale=scale2unity(model.geom_size[geom_id].tolist(), visual_type),
        )

        mesh = None
        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh = self._process_mesh(model, int(model.geom_dataid[geom_id]))

        material = create_material(model.geom_rgba[geom_id].tolist())
        mat_id = int(model.geom_matid[geom_id])
        if mat_id != -1:
            material = self._process_material(model, mat_id)

        return SimVisual(
            name=geom_name, type=visual_type, mesh=mesh, material=material, trans=trans
        )

    def _process_mesh(self, model, mesh_id: int) -> SimMesh:
        vert_adr = int(model.mesh_vertadr[mesh_id])
        vert_num = int(model.mesh_vertnum[mesh_id])
        face_adr = int(model.mesh_faceadr[mesh_id])
        face_num = int(model.mesh_facenum[mesh_id])
        verts = model.mesh_vert[vert_adr : vert_adr + vert_num].astype(np.float32)
        faces = model.mesh_face[face_adr : face_adr + face_num].astype(np.int32)

        # Manual unmerge: one vertex per face-corner, ordered exactly like
        # faces.flatten(). UVs computed in this same order line up row-for-row and
        # are never reshuffled (the bug in simpub's unmerge_vertices path).
        unmerged_verts = verts[faces].reshape(-1, 3)
        unmerged_faces = np.arange(len(unmerged_verts), dtype=np.int32).reshape(-1, 3)

        uv = None
        uv_adr = int(model.mesh_texcoordadr[mesh_id])
        if uv_adr != -1:
            uv_num = int(model.mesh_texcoordnum[mesh_id])
            all_uv = model.mesh_texcoord[uv_adr : uv_adr + uv_num]
            face_uv_idx = model.mesh_facetexcoord[face_adr : face_adr + face_num]
            uv = all_uv[face_uv_idx].reshape(-1, 2).astype(np.float32)
            # MuJoCo/OpenGL texture origin is bottom-left; Unity samples top-left.
            uv[:, 1] = 1.0 - uv[:, 1]

        # Consistent outward normals; fix_normals may flip face winding (vertex
        # array is unchanged, so the UV alignment is preserved).
        tm = trimesh.Trimesh(
            vertices=unmerged_verts, faces=unmerged_faces, process=False
        )
        tm.fix_normals()
        out_verts = np.asarray(tm.vertices, dtype=np.float32)
        out_faces = np.asarray(tm.faces, dtype=np.int32)
        out_normals = np.asarray(tm.vertex_normals, dtype=np.float32)

        return SimMesh(
            vertices=self._encode_vec3(out_verts),
            normals=self._encode_vec3(out_normals),
            # reverse winding to compensate the orientation-reversing remap
            indices=np.ascontiguousarray(
                out_faces[:, [2, 1, 0]], dtype=np.int32
            ).tobytes(),
            uv=np.ascontiguousarray(uv, dtype=np.float32).tobytes()
            if uv is not None
            else None,
        )

    @staticmethod
    def _encode_vec3(arr: np.ndarray) -> bytes:
        """mj -> unity remap [x,y,z] -> [-y, z, x], then row-major float32 bytes."""
        out = arr[:, [1, 2, 0]].astype(np.float32, copy=True)
        out[:, 0] = -out[:, 0]
        return np.ascontiguousarray(out).tobytes()

    def _process_material(self, model, mat_id: int):
        tex_id = model.mat_texid[mat_id]
        if isinstance(tex_id, np.ndarray):
            tex_id = int(tex_id[1])  # mjTEXROLE_RGB (albedo) in mujoco 3.x
        elif isinstance(tex_id, (np.integer, int)):
            tex_id = int(tex_id)
        else:
            logger.warning(
                "Unsupported mat_texid type %s; ignoring texture.", type(tex_id)
            )
            tex_id = -1

        texture = None
        if tex_id != -1:
            texture = self._process_texture(model, tex_id)
            texture["textureScale"] = model.mat_texrepeat[mat_id].tolist()

        return create_material(
            color=model.mat_rgba[mat_id].tolist(),
            emissionColor=[0.0, 0.0, 0.0, 1.0],
            specular=float(model.mat_specular[mat_id]),
            shininess=float(model.mat_shininess[mat_id]),
            reflectance=float(model.mat_reflectance[mat_id]),
            texture=texture,
        )

    def _process_texture(self, model, tex_id: int):
        height = int(model.tex_height[tex_id])
        width = int(model.tex_width[tex_id])
        nchannel = (
            int(model.tex_nchannel[tex_id]) if hasattr(model, "tex_nchannel") else 3
        )
        assert nchannel == 3, "Only 3-channel textures are supported."
        adr = int(model.tex_adr[tex_id])
        num = height * width * nchannel
        tex_data = model.tex_data if hasattr(model, "tex_data") else model.tex_rgb
        return create_texture(tex_data[adr : adr + num], height, width)
