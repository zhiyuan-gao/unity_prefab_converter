#!/usr/bin/env python3

import os
import re
import json
from typing import Dict, Any, List
import numpy
from scipy.spatial.transform import Rotation
from multiverse_parser import Configuration, Factory
from multiverse_parser import (WorldBuilder,
                               BodyBuilder,
                               GeomType, GeomProperty, 
                               MaterialProperty,MaterialBuilder,
                               MeshProperty)
from multiverse_parser import MjcfExporter, UrdfExporter

from pxr import Usd, UsdGeom, UsdShade, Gf
import random
import argparse

from create_thick_wall import create_wall_info, sort_rectangle_vertices,triangulate_2d

source_dir = os.path.dirname(os.path.realpath(__file__))


def snake_to_camel(snake_str):
    # Split the string by underscores, but keep numbers separated
    components = re.split(r'(_\d+_)|(_\d+)|_', snake_str)

    # Filter out None or empty strings that might result from the split
    components = [comp for comp in components if comp]

    # Capitalize the first letter of each component except those that are purely numeric
    camel_str = ''.join(comp.title() if not comp.isdigit() else comp for comp in components)

    camel_str = re.sub(r'(\d+)', r'_\1_', camel_str)

    camel_str = camel_str.strip('_').replace('.', '_').replace(' ', '_').replace('__', '_')

    return camel_str


def get_asset_paths(asset_name: str) -> List[str]:
    asset_name = asset_name.replace("Bathroom", "").replace("Photo", "").replace("Painting", "")
    print("Importing asset:", asset_name)
    asset_path = os.path.join(source_dir, "grp_objects", asset_name)

    asset_paths = []
    if os.path.exists(asset_path):
        asset_path = os.path.join(asset_path, f"{asset_name}.stl")
        if not os.path.exists(asset_path):
            raise FileNotFoundError("File not found:", asset_path)
        asset_paths.append(asset_path)
        return asset_paths
    else:
        for root, dirs, files in os.walk(os.path.join(source_dir, "single_objects")):
            for file in files:
                file_name = os.path.splitext(file)[0]
                if file.endswith('.stl') and asset_name in file_name:
                    asset_path = os.path.join(root, file)
                    if os.path.exists(asset_path):
                        asset_paths.append(asset_path)
        if len(asset_paths) > 0:
            return asset_paths

        asset_new_name = re.sub(r'_\d+$', '', asset_name)
        if asset_new_name[-1] == '_':
            asset_new_name = asset_new_name[:-1]
        if asset_new_name != asset_name:
            print(f"Asset not found: {asset_name}, try to remove the last numbers")
            return get_asset_paths(asset_new_name)

        for root, dirs, files in os.walk(os.path.join(source_dir, "grp_objects")):
            for file in files:
                file_name = os.path.splitext(file)[0]
                if file.endswith('.stl') and asset_name in file_name:
                    asset_path = os.path.join(root, file)
                    if os.path.exists(asset_path):
                        asset_paths.append(asset_path)
        if len(asset_paths) > 0:
            return asset_paths

        for root, dirs, files in os.walk(os.path.join(source_dir, "grp_objects")):
            for file in files:
                file_name = os.path.splitext(file)[0]
                if file.endswith('.stl') and file_name in asset_name:
                    asset_path = os.path.join(root, file)
                    if os.path.exists(asset_path):
                        asset_paths.append(asset_path)
        if len(asset_paths) > 0:
            return asset_paths

        for root, dirs, files in os.walk(os.path.join(source_dir, "single_objects")):
            for file in files:
                file_name = os.path.splitext(file)[0]
                if file.endswith('.stl') and file_name in asset_name:
                    asset_path = os.path.join(root, file)
                    if os.path.exists(asset_path):
                        asset_paths.append(asset_path)
        if len(asset_paths) > 0:
            return asset_paths

    if asset_name[-1].isupper():
        print(f"Asset not found: {asset_name}, try to remove the last capital character")
        asset_new_name = asset_name[:-1]
        return get_asset_paths(asset_new_name)

    return asset_paths


class ProcthorImporter(Factory):
    def __init__(self, file_path: str, config: Configuration):
        super().__init__(file_path, config)
        with open(file_path) as f:
            self.house = json.load(f)

        self._world_builder = WorldBuilder(usd_file_path=self.tmp_usd_file_path)

        with open('AllPrefabDetails.json', 'r') as file:
            self.all_prefab_details = json.load(file)

        body_builder = self._world_builder.add_body(body_name=house_name)

        objects = self.house["objects"]
        for obj in objects:
            self.import_object(house_name, obj)

        walls = self.house["walls"]
        doors = self.house["doors"]
        windows = self.house["windows"]
        rooms= self.house["rooms"]

        for window_id, window in enumerate(windows):
            self.import_hole_cover(window, window_id, walls)

        for door_id, door in enumerate(doors):
            self.import_hole_cover(door, door_id, walls)


        structure = create_wall_info(self.house, material_db={},)

        for wall_id, wall in enumerate(structure["walls"]):

            self.import_wall(wall, wall_id)
        
        for room_id, room in enumerate(rooms):
            self.import_floor(room, room_id)


    def import_object(self, parent_body_name: str, obj: Dict[str, Any]) -> None:

        body_name = obj["id"].replace("|", "_").replace("_surface", "")
        body_builder = self._world_builder.add_body(body_name=body_name, parent_body_name=house_name)

        position = obj.get("position", {"x": 0, "y": 0, "z": 0})
        position_vec = numpy.array([position["x"], position["y"], position["z"]])
        rotation = obj.get("rotation", {"x": 0, "y": 0, "z": 0})
        rotation_mat = Rotation.from_euler("xyz", [rotation["x"], rotation["y"], rotation["z"]],
                                           degrees=True)

        x_90_rotation_matrix = numpy.array([[1, 0, 0],
                                            [0, 0, -1],
                                            [0, 1, 0]])
        
        asset_id = obj["assetId"]

        position_vec = numpy.dot(x_90_rotation_matrix, position_vec)
        rotation_quat = Rotation.from_matrix(
            numpy.dot(x_90_rotation_matrix, numpy.dot(rotation_mat.as_matrix(), x_90_rotation_matrix.T))).as_quat()

        body_builder.set_transform(pos=position_vec, quat=rotation_quat)

        if "assetId" not in obj:
            return None

        self.import_asset(body_builder, asset_id)

        for child in obj.get("children", {}):
            self.import_object(body_name, child)


    def import_wall(self, wall: Dict[str, Any], wall_id: int) -> None:
        

        vertex = wall['vertices']
        vertex = sort_rectangle_vertices(vertex)

        thickness = wall['thickness']

        body_name = f"Wall_{wall_id}"
        body_builder = self._world_builder.add_body(body_name=body_name, parent_body_name=house_name)

        x_90_rotation_matrix = numpy.array([[1, 0, 0],
                                            [0, 0, -1],
                                            [0, 1, 0]])
        rotation_quat = Rotation.from_matrix(x_90_rotation_matrix).as_quat()

        body_builder.set_transform(quat=rotation_quat)

        constant_dim = numpy.where(numpy.all(vertex == vertex[0, :], axis=0))[0][0]

        back_vertex = vertex.copy()
        front_vertex = vertex.copy()
        half_thickness = thickness / 2
        front_vertex[:, constant_dim] -= half_thickness
        back_vertex[:, constant_dim] += half_thickness

        sorted_points = numpy.vstack((front_vertex, back_vertex))

        face_vertex_counts = numpy.array([3] * 12)

        face_vertex_indices = numpy.array([
            0, 2, 1, 0, 3, 2,  # front
            4, 5, 6, 4, 6, 7,  # back
            0, 1, 5, 0, 5, 4,  # left
            3, 6, 2, 3, 7, 6,  # right
            1, 2, 6, 1, 6, 5,  # top
            0, 4, 7, 0, 7, 3   # bottom
        ])

        # normals
        normals = numpy.array([
            [0, 0, 1], [0, 0, 1], [0, 0, 1],
            [0, 0, 1], [0, 0, 1], [0, 0, 1],  # front

            [0, 0, -1], [0, 0, -1], [0, 0, -1],
            [0, 0, -1], [0, 0, -1], [0, 0, -1],  # back

            [-1, 0, 0], [-1, 0, 0], [-1, 0, 0],
            [-1, 0, 0], [-1, 0, 0], [-1, 0, 0],  # left

            [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [1, 0, 0], [1, 0, 0], [1, 0, 0],  # right

            [0, 1, 0], [0, 1, 0], [0, 1, 0],
            [0, 1, 0], [0, 1, 0], [0, 1, 0],  # top

            [0, -1, 0], [0, -1, 0], [0, -1, 0],
            [0, -1, 0], [0, -1, 0], [0, -1, 0]  # bottom
        ])

        # NOTE: Doors and windows cannot share the same wall, per AI2-THOR limitation.Therefore, there can be at most one hole in the wall
        # To create convex mesh, we need to split the wall with holes into rectangles. So we need extra vertices

        if vertex.shape[0] == 12:

            points_mesh_top = numpy.array([sorted_points[8],sorted_points[1],sorted_points[4],sorted_points[9],
                        sorted_points[20],sorted_points[13],sorted_points[16],sorted_points[21]])
    
            points_mesh_left = numpy.array([sorted_points[10],sorted_points[8],sorted_points[3],sorted_points[2],
                        sorted_points[22],sorted_points[20],sorted_points[15],sorted_points[14]])
            
            points_mesh_right = numpy.array([sorted_points[7],sorted_points[5],sorted_points[9],sorted_points[11],
                        sorted_points[19],sorted_points[17],sorted_points[21],sorted_points[23]])
            
            points_mesh_bottom = numpy.array([sorted_points[0],sorted_points[10],sorted_points[11],sorted_points[6],
                        sorted_points[12],sorted_points[22],sorted_points[23],sorted_points[18]])
            
            points_list = [points_mesh_top, points_mesh_left, points_mesh_right, points_mesh_bottom]

            for idx, points in enumerate(points_list):
                mesh_file_name = f"Wall_{wall_id}_{idx}"

                mesh_property = MeshProperty(points=points,
                                             normals=normals,
                                             face_vertex_counts=face_vertex_counts,
                                             face_vertex_indices=face_vertex_indices,
                                             mesh_file_name=mesh_file_name)
                geom_property = GeomProperty(geom_type=GeomType.MESH,
                                             is_visible=True,
                                             is_collidable=True)
                geom_builder = body_builder.add_geom(geom_name=f"{body_name}_{idx}",
                                                     geom_property=geom_property)
                geom_builder.add_mesh(mesh_name=f"SM_{body_name}_{idx}", mesh_property=mesh_property)

        elif vertex.shape[0] == 10:
            points_mesh_top = numpy.array([sorted_points[8],sorted_points[1],sorted_points[4],sorted_points[9],
                        sorted_points[18],sorted_points[11],sorted_points[14],sorted_points[19]])
            
            points_mesh_left = numpy.array([sorted_points[0],sorted_points[8],sorted_points[3],sorted_points[2],
                        sorted_points[10],sorted_points[18],sorted_points[13],sorted_points[12]])
            
            points_mesh_right = numpy.array([sorted_points[7],sorted_points[5],sorted_points[9],sorted_points[6],
                        sorted_points[17],sorted_points[15],sorted_points[19],sorted_points[16]])
            
            points_list = [points_mesh_top, points_mesh_left, points_mesh_right]

            for idx, points in enumerate(points_list):
                mesh_file_name = f"Wall_{wall_id}_{idx}"

                mesh_property = MeshProperty(points=points,
                                             normals=normals,
                                             face_vertex_counts=face_vertex_counts,
                                             face_vertex_indices=face_vertex_indices,
                                             mesh_file_name=mesh_file_name)
                geom_property = GeomProperty(geom_type=GeomType.MESH,
                                             is_visible=True,
                                             is_collidable=True)
                geom_builder = body_builder.add_geom(geom_name=f"{body_name}_{idx}",
                                                     geom_property=geom_property)
                geom_builder.add_mesh(mesh_name=f"SM_{body_name}_{idx}", mesh_property=mesh_property)


        elif vertex.shape[0] == 4:
            mesh_file_name = f"Wall_{wall_id}"

            mesh_property = MeshProperty(points=sorted_points,
                                         normals=normals,
                                         face_vertex_counts=face_vertex_counts,
                                         face_vertex_indices=face_vertex_indices,
                                         mesh_file_name=mesh_file_name)
            geom_property = GeomProperty(geom_type=GeomType.MESH,
                                         is_visible=True,
                                         is_collidable=True)
            geom_builder = body_builder.add_geom(geom_name=f"{body_name}",
                                                 geom_property=geom_property)
            geom_builder.add_mesh(mesh_name=f"SM_{body_name}", mesh_property=mesh_property)

    def import_floor(self, room: Dict[str, Any], room_id: int) -> None:
        """
        Import a floor by extruding the floor polygon into a volume.
        The floor's top surface will be at 0 height, and the bottom at -thickness.
        """

        body_name = f"Floor_{room_id}"
        body_builder = self._world_builder.add_body(body_name=body_name, parent_body_name=house_name)

        x_90_rotation_matrix = numpy.array([[1, 0, 0],
                                            [0, 0, -1],
                                            [0, 1, 0]])
        rotation_quat = Rotation.from_matrix(x_90_rotation_matrix).as_quat()

        body_builder.set_transform(quat=rotation_quat)

        floorPolygon = room['floorPolygon']
        # Build a list of 2D points (using x and z) for triangulation
        points2d = [(v["x"], v["z"]) for v in floorPolygon]
        # Get triangulated indices for the flat polygon (flat list, every three numbers form a triangle)
        bottom_tri = triangulate_2d(points2d, clockWise=False)
        n = len(points2d)
        
        # Build vertex lists for bottom and top faces
        # Combine vertices for both faces: indices 0 ~ n-1 for bottom, n ~ 2n-1 for top
        top_vertices = [Gf.Vec3f(v["x"], v["y"], v["z"]) for v in floorPolygon]
        bottom_vertices = [Gf.Vec3f(v["x"], v["y"] - 0.1, v["z"]) for v in floorPolygon]
        vertices = top_vertices + bottom_vertices

        # Bottom face: use the triangulation result, order should ensure correct normal direction 
        bottom_face_indices = bottom_tri  # The bottom face normal points downward (reverse if needed)
        
        # Top face: use the same triangulation result, but add n and reverse order to ensure upward normal
        top_face_indices = []
        for j in range(0, len(bottom_tri), 3):
            a = bottom_tri[j] + n
            b = bottom_tri[j + 1] + n
            c = bottom_tri[j + 2] + n
            top_face_indices.extend([c, b, a])
        
        # Sides: for each edge of the polygon, create two triangles forming a quad
        side_indices = []
        for j in range(n):
            j_next = (j + 1) % n
            # Quad vertex order: bottom j, bottom j_next, top j_next, top j
            # Split into two triangles:
            side_indices.extend([j, j_next, j_next + n])
            side_indices.extend([j, j_next + n, j + n])
        
        # Combine all triangle indices
        all_triangles = bottom_face_indices + top_face_indices + side_indices

        numTriangles = len(all_triangles) // 3
        faceVertexCounts = [3] * numTriangles
  
        mesh_file_name = f"SM_Floor_{room_id}"

        vertices_array = numpy.array([ (v[0], v[1], v[2]) for v in vertices ], dtype=float)

        # Convert face vertex counts and indices to numpy arrays
        face_vertex_counts_array = numpy.array(faceVertexCounts, dtype=int)
        face_vertex_indices_array = numpy.array(all_triangles, dtype=int)

        mesh_property = MeshProperty(points=vertices_array,
                                    normals=None,  # Optionally compute normals if needed
                                    face_vertex_counts=face_vertex_counts_array,
                                    face_vertex_indices=face_vertex_indices_array,
                                    mesh_file_name=mesh_file_name)

        geom_property = GeomProperty(geom_type=GeomType.MESH,
                                    is_visible=True,
                                    is_collidable=True)

        geom_builder = body_builder.add_geom(geom_name=body_name, geom_property=geom_property)
        geom_builder.add_mesh(mesh_name=f"SM_{body_name}", mesh_property=mesh_property)


    def import_hole_cover(self, hole_cover: Dict[str, Any],hole_cover_id: int, walls: List[Dict[str, Any]]) -> None:

        hole_cover_class = hole_cover['id'].split('|')[0]
        wall0_id = hole_cover['wall0']
        wall1_id = hole_cover['wall1']

        for wall in walls:
            if wall["id"] == wall1_id:
                break
        else:
            raise ValueError(f"Wall {wall1_id} not found")

        for wall in walls:
            if wall["id"] == wall0_id:
                poly_wall0 = wall
                break
        else:
            raise ValueError(f"Wall {wall0_id} not found")


        wall0 = polygon_wall_to_simple_wall(poly_wall0, hole_cover)

        p1 = numpy.array([wall0['p1']['x'], wall0['p1']['y'], wall0['p1']['z']])
        p0 = numpy.array([wall0['p0']['x'], wall0['p0']['y'], wall0['p0']['z']])
        p0p1 = p1 - p0
        p0p1_norm = p0p1 / numpy.linalg.norm(p0p1)

        position_vec = p0 + (p0p1_norm * (hole_cover['assetPosition']['x'])) + numpy.array([0, 1, 0])* hole_cover['assetPosition']['y']

        theta = -numpy.sign(p0p1_norm[2]) * numpy.arccos(numpy.dot(p0p1_norm, numpy.array([1, 0, 0])))
        rotY = numpy.degrees(theta)

        # somehow need a extra 180 degree rotation around z axis
        rotation_mat = Rotation.from_euler("xyz", [0, rotY+180, 0],
                                           degrees=True)

        x_90_rotation_matrix = numpy.array([[1, 0, 0],
                                            [0, 0, -1],
                                            [0, 1, 0]])
        
        if hole_cover_class == 'door':
            body_name = f"Door_{hole_cover_id}"
        elif hole_cover_class == 'window':
            body_name = f"Window_{hole_cover_id}"
        else:
            raise ValueError(f"Unknown hole cover class: {hole_cover_class}")

        body_builder = self._world_builder.add_body(body_name=body_name, parent_body_name=house_name)

        position_vec = numpy.dot(x_90_rotation_matrix, position_vec)
        rotation_quat = Rotation.from_matrix(
            numpy.dot(x_90_rotation_matrix, numpy.dot(rotation_mat.as_matrix(), x_90_rotation_matrix.T))).as_quat()

        body_builder.set_transform(pos=position_vec, quat=rotation_quat)

        asset_name = hole_cover["assetId"]
        self.import_asset(body_builder, asset_name)

    def import_asset(self, body_builder: BodyBuilder, asset_name: str) -> None:
        # asset_paths = get_asset_paths(asset_name)
        asset_paths = '/home/zgao/unity_prefab_converter/procthor_assets'
        prefab_info = self.all_prefab_details[asset_name]

        folder_path = os.path.join(asset_paths, asset_name)
        asset_name = re.sub(r'(\d+)x(\d+)', r'\1_X_\2', asset_name)
        

        obj_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.obj')]
        
        mesh_idx = 0

        for asset_path in obj_files:

            body_name = body_builder.xform.GetPrim().GetName()
            tmp_usd_mesh_file_path, tmp_origin_mesh_file_path = self.import_mesh(
                mesh_file_path=asset_path, merge_mesh=True)

            mesh_stage = Usd.Stage.Open(tmp_usd_mesh_file_path)
            for mesh_prim in [prim for prim in mesh_stage.Traverse() if prim.IsA(UsdGeom.Mesh)]:
                mesh_name = mesh_prim.GetName()
                mesh_path = mesh_prim.GetPath()
                mesh_property = MeshProperty.from_mesh_file_path(mesh_file_path=tmp_usd_mesh_file_path,
                                                                 mesh_path=mesh_path)
                geom_property = GeomProperty(geom_type=GeomType.MESH,
                                             is_visible=True,
                                             is_collidable=True)
                geom_builder = body_builder.add_geom(geom_name=f"SM_{body_name}_{asset_name}_{mesh_idx}",
                                                     geom_property=geom_property)
                
                geom_builder.add_mesh(mesh_name=mesh_name, mesh_property=mesh_property)


                if mesh_prim.HasAPI(UsdShade.MaterialBindingAPI):
                    material_binding_api = UsdShade.MaterialBindingAPI(mesh_prim)
                    material_paths = material_binding_api.GetDirectBindingRel().GetTargets()
                    if len(material_paths) > 1:
                        raise NotImplementedError(f"Mesh {body_name} has more than one material.")
                    material_path = material_paths[0]
                    material_property = MaterialProperty.from_material_file_path(
                        material_file_path=tmp_usd_mesh_file_path,
                        material_path=material_path)
                    if material_property.opacity == 0.0:
                        print(f"Opacity of {material_path} is 0.0. Set to 1.0.")
                        material_property._opacity = 1.0
                    material_builder = geom_builder.add_material(material_name=material_path.name,
                                                                    material_property=material_property)



                mesh_idx += 1




def polygon_wall_to_simple_wall(wall, holes):

    polygons = sorted(wall['polygon'], key=lambda p: p['y'])
    max_y = max(p['y'] for p in wall['polygon'])
    
    hole = holes.get(wall['id'], None)
    
    p0 = polygons[0]
    
    return {
        'id': wall['id'],
        'p0': polygons[0],
        'p1': polygons[1],
        'height': max_y - p0['y'],
        'material': wall['material'],
        'roomId': wall['roomId'],

    }



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Auto semantic tagging based on object names")
    parser.add_argument("--house", type=str, required=True, help="Input JSON")
    args = parser.parse_args()

    house_name = f"house_{args.house}"

    house_file_path = os.path.join(source_dir, f"{house_name}.json")
    config = Configuration()
    factory = ProcthorImporter(file_path=house_file_path, config=config)

    # Export to USD
    house_usd_file_path = os.path.join(source_dir, house_name, f"{house_name}.usda")
    factory.save_tmp_model(house_usd_file_path)

    # Export to URDF and MJCF
    urdf_exporter = UrdfExporter(file_path=house_usd_file_path.replace(".usda", ".urdf"), factory=factory)
    urdf_exporter.build()
    urdf_exporter.export(keep_usd=False)

    mjcf_exporter = MjcfExporter(file_path=house_usd_file_path.replace(".usda", ".xml"), factory=factory)
    mjcf_exporter.build()
    mjcf_exporter.export(keep_usd=False)