from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
import numpy as np
from scipy.spatial.transform import Rotation
import os

class Wall:
    def __init__(self, id, p0, p1, height, thickness, material, empty, roomId, hole=None, layer=None):
        self.id = id
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)
        self.height = height
        self.thickness = thickness
        self.material = material
        self.empty = empty
        self.roomId = roomId
        self.hole = hole
        self.layer = layer

class PolygonWall:
    def __init__(self, id, polygon, material, empty, roomId, thickness=0.1, layer=None):
        self.id = id
        self.polygon = [np.array(point) for point in polygon]  # List of points defining the polygon
        self.material = material
        self.empty = empty
        self.roomId = roomId
        self.thickness = thickness
        self.layer = layer

class WallRectangularHole:
    def __init__(self, id, asset_id, room0, room1, wall0, wall1, hole_polygon, asset_position,  scale=None, material=None):
        self.id = id
        self.asset_id = asset_id
        self.room0 = room0
        self.room1 = room1
        self.wall0 = wall0
        self.wall1 = wall1
        self.hole_polygon = [np.array(point) for point in hole_polygon]
        # print(self.hole_polygon)
        self.asset_position = np.array(asset_position)
        # self.openness = openness
        self.scale = np.array(scale) if scale else None
        self.material = material

class BoundingBox:
    def __init__(self, min_point, max_point):
        # 由于min_point和max_point是numpy数组，且包含字典，需要先将字典提取出来
        min_point_dict = min_point.item() if isinstance(min_point, np.ndarray) else min_point
        max_point_dict = max_point.item() if isinstance(max_point, np.ndarray) else max_point

        # 提取字典中的坐标值并转换为 numpy 数组
        self.min = np.array([min_point_dict['x'], min_point_dict['y'], min_point_dict['z']])
        self.max = np.array([max_point_dict['x'], max_point_dict['y'], max_point_dict['z']])

    def center(self):
        return self.min + (self.max - self.min) / 2.0

    def size(self):
        return self.max - self.min
    
def polygon_wall_to_simple_wall(wall, holes):
    # 提取 y 坐标并排序
    sorted_polygon = sorted(wall.polygon, key=lambda p: p.item()['y'])

    # 找到最大 y 坐标
    max_y = max(p.item()['y'] for p in wall.polygon)

    # 查找对应的 hole
    hole = holes.get(wall.id, None)

    # 获取排序后的前两个点
    p0 = sorted_polygon[0].item()
    p1 = sorted_polygon[1].item()

    return Wall(
        id=wall.id,
        p0=np.array([p0['x'], p0['y'], p0['z']]),
        p1=np.array([p1['x'], p1['y'], p1['z']]),
        height=max_y - p0['y'],  
        thickness=wall.thickness,
        material=wall.material,
        empty=wall.empty,
        roomId=wall.roomId,
        hole=hole,
        layer=wall.layer
    )



def generate_holes(house):
    windows_and_doors = house['doors'] + house['windows']

    holes = {}
    for hole in windows_and_doors:
        hole_obj = WallRectangularHole(
            id=hole['id'],
            asset_id=hole['assetId'],
            room0=hole['room0'],
            room1=hole['room1'],
            wall0=hole['wall0'],
            wall1=hole['wall1'],
            hole_polygon=hole['holePolygon'],
            asset_position=hole['assetPosition'],
            # openness=hole['openness'],
            scale=hole.get('scale'),
            material=hole.get('material')
        )
        if hole_obj.wall0:
            holes[hole_obj.wall0] = hole_obj
        if hole_obj.wall1:
            holes[hole_obj.wall1] = hole_obj
    return holes

def generate_wall_vertices(to_create, back_faces=False):
    p0p1 = np.array(to_create.p1) - np.array(to_create.p0)
    p0p1_norm = p0p1 / np.linalg.norm(p0p1)

    width = np.linalg.norm(p0p1)
    height = to_create.height

    p0 = to_create.p0
    p1 = to_create.p1
    # 计算 theta
    # theta = -np.sign(p0p1_norm[2]) * np.arccos(np.dot(p0p1_norm, np.array([1, 0, 0])))
    # print('theta',theta)

    vertices = []
    triangles = []

    if to_create.hole:
        hole_bb = get_hole_bounding_box(to_create.hole)
        dims = hole_bb.size()
        offset = [hole_bb.min[0], hole_bb.min[1]]

        if to_create.hole.wall1 == to_create.id:
            offset = [width - hole_bb.max[0], hole_bb.min[1]]

        vertices = [
            p0,
            p0 + np.array([0.0, height, 0.0]),
            p0 + p0p1_norm * offset[0] + np.array([0.0, offset[1], 0.0]),
            p0 + p0p1_norm * offset[0] + np.array([0.0, offset[1] + dims[1], 0.0]),
            p1 + np.array([0.0, height, 0.0]),
            p0 + p0p1_norm * (offset[0] + dims[0]) + np.array([0.0, offset[1] + dims[1], 0.0]),
            p1,
            p0 + p0p1_norm * (offset[0] + dims[0]) + np.array([0.0, offset[1], 0.0])
        ]


        triangles = [
            0, 1, 2, 1, 3, 2, 1, 4, 3, 3, 4, 5, 4, 6, 5, 5, 6, 7, 7, 6, 0, 0, 2, 7
        ]

        if back_faces:
            triangles.extend([t for t in reversed(triangles)])
    else:
        vertices = [
            p0.tolist(),
            [p0[0], p0[1] + height, p0[2]],
            [p1[0], p1[1] + height, p1[2]],
            p1.tolist()
        ]

        triangles = [1, 2, 0, 2, 3, 0]

        if back_faces:
            triangles.extend([t for t in reversed(triangles)])

    return vertices, triangles

def get_hole_bounding_box(hole):
    if hole.hole_polygon is None or len(hole.hole_polygon) < 2:
        raise ValueError(f"Invalid `holePolygon` for object id: '{hole.id}'. Minimum 2 vertices indicating first min and second max of hole bounding box.")
    return BoundingBox(min_point=hole.hole_polygon[0], max_point=hole.hole_polygon[1])

def create_wall_info(house, material_db, game_object_id="Structure"):
    holes = generate_holes(house)

    structure = {"id": game_object_id, "walls": []}

    ignore_walls = []

    # Convert each wall dictionary to a PolygonWall object, and set a default thickness if missing
    walls = [PolygonWall(
                id=w['id'],
                polygon=w['polygon'],
                material=w.get('material'),
                empty=w.get('empty', False),
                roomId=w['roomId'],
                # thickness=w.get('thickness', 0.1),  # defualt 0.1
                layer=w.get('layer')
            ) for w in house['walls']]

    walls = [polygon_wall_to_simple_wall(w, holes) for w in walls]

    walls_per_room = {}
    for wall in walls:
        walls_per_room.setdefault(wall.roomId, []).append(wall)

    zip3 = []
    for room_walls in walls_per_room.values():
        room_zip3 = []
        n = len(room_walls)
        for i in range(n):
            w0 = room_walls[i]
            w1 = room_walls[(i + 1) % n]
            w2 = room_walls[(i - 1) % n]
            room_zip3.append((w0, w1, w2))
        zip3.append(room_zip3)

    index = 0
    for wall_tuples in zip3:
        for w0, w1, w2 in wall_tuples:
            if not w0.empty:
                vertices, triangles = generate_wall_vertices(
                    w0,
                )
                wall_go = {
                    "index": index,
                    "id": w0.id,
                    "vertices": vertices,
                    "triangles": triangles,
                    "thickness": w0.thickness
                }
                wall_id_str = ' '.join(wall_go["id"].split('|')[-4:])

                if wall_id_str in ignore_walls:
                    continue
                ignore_walls.append(wall_id_str)
                structure["walls"].append(wall_go)
                index += 1

    return structure


def sort_rectangle_vertices(points):
    """
    Sorts the eight vertices into two rectangles, by removing the constant dimension,
    sorting the 2D points, and then restoring the third dimension.

    :param points: List or array of eight vertices (each a tuple of (x, y, z)).
    :return: List of sorted vertices in the specified order with the original dimensions restored.
    """
    points = np.array(points)

    # Find the dimension that is constant (i.e., all points have the same value in this dimension)
    constant_dim = np.where(np.all(points == points[0, :], axis=0))[0][0]

    reduced_points = np.delete(points, constant_dim, axis=1)


    if constant_dim == 0:
        reduced_points[:, [0, 1]] = reduced_points[:, [1, 0]]


    # Apply the previous sorting logic to the 2D points
    sorted_reduced_points = sort_2d_rectangle_vertices(reduced_points)
    # print(type(sorted_reduced_points))
    sorted_reduced_points = np.array(sorted_reduced_points)

    if constant_dim == 0:
        sorted_reduced_points[:, [0, 1]] = sorted_reduced_points[:, [1, 0]]
    
    # Restore the removed dimension
    sorted_points = np.insert(sorted_reduced_points, constant_dim, points[0, constant_dim], axis=1)


    return sorted_points

def split_non_convex_wall(wall):
    pass


def sort_2d_rectangle_vertices(points: np.ndarray) -> np.ndarray:
    """
    Sorts vertices into two rectangles, with specific ordering:
    if points has 8 vertices, 4 vertices for each rectangle
    Large rectangle (0: bottom-left, 1: top-left, 4: top-right, 6: bottom-right)
    Small rectangle (2: bottom-left, 3: top-left, 5: top-right, 7: bottom-right)
    Vertices 0 - 7 are sorted as the same order as it in ai2thor
    8-11 are the projection of the door or window on the wall
    8: pro_up_left, 9: pro_up_right, 10: pro_bottom_left, 11: pro_bottom_right

    In order to split the wall with holes into rectangles, add two extra vertices for walls 
    with a door and four extra vertices for walls with a window

    if points has 4 vertices
    0: bottom-left, 1: top-left, 2: top-right, 3: bottom-right.

    :param points: List of 8 points (x, y) representing two rectangles.
    :return: np.array of sorted points.
    """

    if points.shape[0] == 8:

        xmin = points[:, 0].min()
        xmax = points[:, 0].max()
        ymin = points[:, 1].min()
        ymax = points[:, 1].max()
        # # Identify the four corners of the large rectangle in 3D
        large_rect_points = [
            (xmin, ymin),
            (xmin, ymax),
            (xmax, ymax),
            (xmax, ymin)
        ]

        # Remove these points from the original list
        small_rect_points = [point for point in points if tuple(point) not in large_rect_points]
    
        # # Function to sort points in the order: bottom-left, top-left, top-right, bottom-right
        def sort_rectangle(pts):
            pts = sorted(pts, key=lambda p: (p[0], p[1]))  # First sort by x, then by y
            return [pts[0], pts[1], pts[3], pts[2]]  # Rearrange to bottom-left, top-left, top-right, bottom-right


        # Sort points within each rectangle
        large_rect_points = sort_rectangle(large_rect_points)

        small_rect_points = sort_rectangle(small_rect_points)

        # projection of the door or window on the wall
        pro_up_left = np.array([large_rect_points[0][0], small_rect_points[1][1]])

        pro_up_right = np.array([large_rect_points[2][0], small_rect_points[2][1]])

        sorted_points = np.array([large_rect_points[0], large_rect_points[1], 
                                small_rect_points[0], small_rect_points[1], 
                                large_rect_points[2], small_rect_points[2], 
                                large_rect_points[3], small_rect_points[3]])
        

        # for doors
        if large_rect_points[0][1] == small_rect_points[0][1]:
            extra_points = np.array([pro_up_left, pro_up_right])

        # for windows
        else:
            pro_bottom_left = np.array([large_rect_points[0][0], small_rect_points[0][1]])
            pro_bottom_right = np.array([large_rect_points[3][0], small_rect_points[3][1]])
            extra_points = np.array([pro_up_left, pro_up_right, pro_bottom_left, pro_bottom_right])

        sorted_points = np.vstack((sorted_points,extra_points))


    elif points.shape[0] == 4:
        
        # Convert the list of vertices to a numpy array for easier manipulation
        vertices = np.array(points)

        # Sort vertices primarily by x, and secondarily by y
        sorted_by_x = vertices[np.argsort(vertices[:, 0])]
        
        # Now, sorted_by_x should have the two left-most points first
        # Separate them into left and right pairs
        left_pair = sorted_by_x[:2]
        right_pair = sorted_by_x[2:]
        
        # Sort the left pair by y to get bottom-left and top-left
        left_pair = left_pair[np.argsort(left_pair[:, 1])]
        
        # Sort the right pair by y to get bottom-right and top-right
        right_pair = right_pair[np.argsort(right_pair[:, 1])]
        
        # Combine them into the desired order: bottom-left, top-left, top-right, bottom-right
        sorted_points = [left_pair[0], left_pair[1], right_pair[1], right_pair[0]]

    else:
        raise ValueError("Invalid number of vertices for a cuboid.")

    return sorted_points

def convert_to_gf_vec3f_list(sorted_points):
    """
    Converts a numpy array of 2D points to a list of Gf.Vec3f objects.

    :param sorted_points: np.array of sorted 2D points.
    :return: List of Gf.Vec3f objects.
    """
    # gf_vec3f_list = [Gf.Vec3f(x, y, z) for x, y, z in sorted_points]
    gf_vec3f_list = [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in sorted_points]


    return gf_vec3f_list

def create_thick_wall(stage, vertex, mesh_prim_path, thickness = 0.1):

    # mesh = UsdGeom.Mesh.Define(stage, mesh_prim_path)
    constant_dim = np.where(np.all(vertex == vertex[0, :], axis=0))[0][0]

    back_vertex = np.copy(vertex)
    back_vertex[:, constant_dim] += thickness

    vertex_vec3f = convert_to_gf_vec3f_list(np.vstack((vertex, back_vertex)))

    faceVertexCounts = [3]*12
    faceVertexIndices = [0, 2, 1, 0, 3, 2,  #front
                        4, 5, 6, 4, 6, 7, #back
                        0, 1, 5, 0, 5, 4, #left
                        3, 6, 2, 3, 7, 6, #right
                        1, 2, 6, 1, 6, 5, #top
                        0, 4, 7, 0, 7, 3  #bottom
                        ]
    normals = [Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), 
                Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), #front

                Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1),
                Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1), #back

                Gf.Vec3f(-1, 0, 0), Gf.Vec3f(-1, 0, 0), Gf.Vec3f(-1, 0, 0),
                Gf.Vec3f(-1, 0, 0), Gf.Vec3f(-1, 0, 0), Gf.Vec3f(-1, 0, 0), #left

                Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 0, 0),
                Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 0, 0),#right

                Gf.Vec3f(0, 1, 0), Gf.Vec3f(0, 1, 0), Gf.Vec3f(0, 1, 0),
                Gf.Vec3f(0, 1, 0), Gf.Vec3f(0, 1, 0), Gf.Vec3f(0, 1, 0),#top

                Gf.Vec3f(0, -1, 0), Gf.Vec3f(0, -1, 0), Gf.Vec3f(0, -1, 0),
                Gf.Vec3f(0, -1, 0), Gf.Vec3f(0, -1, 0), Gf.Vec3f(0, -1, 0),#bottom
    ]

    # NOTE: Doors and windows cannot share the same wall, per AI2-THOR limitation.Therefore, there can be at most one hole in the wall
    # To create convex mesh, we need to split the wall with holes into rectangles. So we need extra vertices

    if vertex.shape[0] == 12:

        points_mesh_top = [vertex_vec3f[8],vertex_vec3f[1],vertex_vec3f[4],vertex_vec3f[9],
                    vertex_vec3f[20],vertex_vec3f[13],vertex_vec3f[16],vertex_vec3f[21]]
   
        points_mesh_left = [vertex_vec3f[10],vertex_vec3f[8],vertex_vec3f[3],vertex_vec3f[2],
                    vertex_vec3f[22],vertex_vec3f[20],vertex_vec3f[15],vertex_vec3f[14]]
        
        points_mesh_right = [vertex_vec3f[7],vertex_vec3f[5],vertex_vec3f[9],vertex_vec3f[11],
                    vertex_vec3f[19],vertex_vec3f[17],vertex_vec3f[21],vertex_vec3f[23]]
        
        points_mesh_bottom = [vertex_vec3f[0],vertex_vec3f[10],vertex_vec3f[11],vertex_vec3f[6],
                    vertex_vec3f[12],vertex_vec3f[22],vertex_vec3f[23],vertex_vec3f[18]]
        
        points_list = [points_mesh_top, points_mesh_left, points_mesh_right, points_mesh_bottom]
        for i, points in enumerate(points_list):
            mesh = UsdGeom.Mesh.Define(stage, f"{mesh_prim_path}_{i}")
            mesh.GetPointsAttr().Set(points)
            mesh.GetFaceVertexCountsAttr().Set(faceVertexCounts)
            mesh.GetFaceVertexIndicesAttr().Set(faceVertexIndices)
            mesh.GetNormalsAttr().Set(normals)
            mesh.SetNormalsInterpolation("faceVarying")

    elif vertex.shape[0] == 10:
        points_mesh_top = [vertex_vec3f[8],vertex_vec3f[1],vertex_vec3f[4],vertex_vec3f[9],
                    vertex_vec3f[18],vertex_vec3f[11],vertex_vec3f[14],vertex_vec3f[19]]
        
        points_mesh_left = [vertex_vec3f[0],vertex_vec3f[8],vertex_vec3f[3],vertex_vec3f[2],
                    vertex_vec3f[10],vertex_vec3f[18],vertex_vec3f[13],vertex_vec3f[12]]
        
        points_mesh_right = [vertex_vec3f[7],vertex_vec3f[5],vertex_vec3f[9],vertex_vec3f[6],
                    vertex_vec3f[17],vertex_vec3f[15],vertex_vec3f[19],vertex_vec3f[16]]
        
        points_list = [points_mesh_top, points_mesh_left, points_mesh_right]

        for i, points in enumerate(points_list):
            mesh = UsdGeom.Mesh.Define(stage, f"{mesh_prim_path}_{i}")
            mesh.GetPointsAttr().Set(points)
            mesh.GetFaceVertexCountsAttr().Set(faceVertexCounts)
            mesh.GetFaceVertexIndicesAttr().Set(faceVertexIndices)
            mesh.GetNormalsAttr().Set(normals)
            mesh.SetNormalsInterpolation("faceVarying")

    elif vertex.shape[0] == 4:

        mesh = UsdGeom.Mesh.Define(stage, mesh_prim_path)
        mesh.GetPointsAttr().Set(vertex_vec3f)
        mesh.GetFaceVertexCountsAttr().Set(faceVertexCounts)
        mesh.GetFaceVertexIndicesAttr().Set(faceVertexIndices)
        mesh.GetNormalsAttr().Set(normals)
        mesh.SetNormalsInterpolation("faceVarying")
        mesh.GetFaceVertexIndicesAttr().Set(faceVertexIndices)


    else:
        raise ValueError("Invalid number of vertices for a cuboid.")


    mesh.GetNormalsAttr().Set(normals)

    mesh.SetNormalsInterpolation("faceVarying")    #could be "vertex", "faceVarying", "uniform", "constant"

    # material_global = UsdShade.Material.Get(stage, "/World/Materials/WhiteMarble__115104")
    # UsdShade.MaterialBindingAPI(mesh).Bind(material_global)

    mesh.GetDoubleSidedAttr().Set(True)



def creat_thick_floor(stage, vertex, mesh_prim_path, transform_matrix, thickness = 0.05):
    pass

if __name__ == "__main__":

    import json
    i = 6

    house_json = f"/home/zgao/unity_prefab_converter/house_{i}.json"
    usd_file_path = f"/home/zgao/unity_prefab_converter/house_{i}/house_{i}.usda"
    new_file_path = f"/home/zgao/unity_prefab_converter/house_{i}/house_{i}_test_wall.usda"

    def process_structure_meshes(house_json , usd_file_path, new_file_path):
        """
        Applies the unique vertex extraction function to all meshes under a specific path in the USD file.

        :param vertices: List of vertices
        :param base_path: Base path in the USD file where meshes are located
        :return: Dictionary containing results for each mesh
        """

        # stage = Usd.Stage.Open(usd_file_path)
        # default_prim = stage.GetDefaultPrim()


        if not os.path.exists(usd_file_path):
            # 如果文件不存在，则创建新的 USD Stage
            stage = Usd.Stage.CreateNew(usd_file_path)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            # 创建一个默认的 Prim，例如一个 Xform
            default_prim = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
            # 设置默认 Prim
            stage.SetDefaultPrim(default_prim)
            # 保存 Stage
            stage.GetRootLayer().Save()
        else:
            # 如果文件存在，打开已有的 Stage
            stage = Usd.Stage.Open(usd_file_path)
            default_prim = stage.GetDefaultPrim()


        # walls_prim = stage.GetPrimAtPath("/World/Structure/Walls")
        with open(house_json) as f:
            house = json.load(f)
            
        structure = create_wall_info(house, material_db={},)

        for i, wall in enumerate(structure["walls"]):

            vertex = wall['vertices']
            thickness = wall['thickness']

            sorted_points = sort_rectangle_vertices(vertex)

            wall_xform = UsdGeom.Xform.Define(stage, f"{default_prim.GetPath()}/Wall_new_{i}")

            wall_xform.AddTranslateOp().Set((0, 0, 0)) 
            wall_xform.AddRotateXOp().Set(90)

            new_mesh_path = f'{default_prim.GetPath()}/Wall_new_{i}/Wall_new_{i}'

            create_thick_wall(stage, sorted_points, new_mesh_path, thickness = thickness)

        # floors_prim = stage.GetPrimAtPath("/World/Structure/Floor")
        # for prim in floors_prim.GetAllChildren():
        #     mesh_path = prim.GetPath().pathString
        #     source_mesh_geom = UsdGeom.Xformable(prim)
        #     transform_matrix = source_mesh_geom.GetLocalTransformation()
        #     vertex = get_unique_vertices_and_indices(usd_file_path, mesh_path)
        #     sorted_points = sort_rectangle_vertices(vertex)
        #     new_mesh_path = mesh_path + "_thick"

        #     create_thick_wall(stage, sorted_points, new_mesh_path, transform_matrix)
        #     stage.RemovePrim(prim.GetPath())


        # ceiling_prim = stage.GetPrimAtPath("/World/Structure/Ceiling")
        # for prim in ceiling_prim.GetAllChildren():
        #     mesh_path = prim.GetPath().pathString
        #     source_mesh_geom = UsdGeom.Xformable(prim)
        #     transform_matrix = source_mesh_geom.GetLocalTransformation()
        #     vertex = get_unique_vertices_and_indices(usd_file_path, mesh_path)
        #     sorted_points = sort_rectangle_vertices(vertex)
        #     new_mesh_path = mesh_path + "_thick"

        #     create_thick_wall(stage, sorted_points, new_mesh_path, transform_matrix)
        #     stage.RemovePrim(prim.GetPath())
        stage.GetRootLayer().Export(new_file_path)

        print("wall mesh has been saved to ", new_file_path)


    process_structure_meshes(house_json, usd_file_path, new_file_path)