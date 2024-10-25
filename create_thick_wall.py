from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
import numpy as np

def get_unique_vertices_and_indices(usd_file_path, mesh_path):
    """
    Extracts unique vertex indices and corresponding vertex coordinates from a given Mesh in a USD file.

    :param usd_file_path: Path to the USD file
    :param mesh_path: Path to the target Mesh in the USD file
    :return: (unique_indices, unique_points, remapped_face_indices)
             - unique_indices: Array of unique vertex indices
             - unique_points: Array of vertex coordinates corresponding to unique_indices
             - remapped_face_indices: Remapped face vertex indices array
    """
    # Load the USD Stage
    stage = Usd.Stage.Open(usd_file_path)

    # Find the specific Mesh
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path))

    # Get vertex indices and vertex coordinates
    face_vertex_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    points = np.array(mesh.GetPointsAttr().Get())

    # Remove duplicate vertex indices
    unique_indices, unique_inverse = np.unique(face_vertex_indices, return_inverse=True)

    # Get the unique vertex coordinates
    unique_points = points[unique_indices]

    # return unique_indices, unique_points, unique_inverse
    return unique_points


def process_structure_meshes(usd_file_path, base_path, new_file_path):
    """
    Applies the unique vertex extraction function to all meshes under a specific path in the USD file.

    :param usd_file_path: Path to the USD file
    :param base_path: Base path in the USD file where meshes are located
    :return: Dictionary containing results for each mesh
    """

    # base_path = "/World/Structure/Walls"
    # Load the USD Stage
    stage = Usd.Stage.Open(usd_file_path)

    walls_prim = stage.GetPrimAtPath("/World/Structure/Walls")

    for prim in walls_prim.GetAllChildren():
        mesh_path = prim.GetPath().pathString
        source_mesh_geom = UsdGeom.Xformable(prim)
        transform_matrix = source_mesh_geom.GetLocalTransformation()
        vertex = get_unique_vertices_and_indices(usd_file_path, mesh_path)
        sorted_points = sort_rectangle_vertices(vertex)
        new_mesh_path = mesh_path + "_thick"

        create_thick_wall(stage, sorted_points, new_mesh_path, transform_matrix)
        stage.RemovePrim(prim.GetPath())

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

    print("thickness mesh has been saved to ", new_file_path)


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

    # Remove the constant dimension to reduce to 2D points
    reduced_points = np.delete(points, constant_dim, axis=1)

    # Apply the previous sorting logic to the 2D points
    sorted_reduced_points = sort_2d_rectangle_vertices(reduced_points)

    # Restore the removed dimension
    sorted_points = np.insert(sorted_reduced_points, constant_dim, points[0, constant_dim], axis=1)

    return sorted_points

def sort_2d_rectangle_vertices(points):
    """
    Sorts vertices into two rectangles, with specific ordering:
    if points has 8 vertices, 4 vertices for each rectangle
    Large rectangle (0: bottom-left, 1: top-left, 4: top-right, 6: bottom-right)
    Small rectangle (2: bottom-left, 3: top-left, 5: top-right, 7: bottom-right)
    if points has 4 vertices
    0: bottom-left, 1: top-left, 2: top-right, 3: bottom-right.

    :param points: List of 8 points (x, y) representing two rectangles.
    :return: np.array of sorted points.
    """

    points = np.array(points)
    # print(points)
    if points.shape[0] == 8:
        # Find the points corresponding to xmin, xmax, ymin, ymax
        xmin = points[:, 0].min()
        xmax = points[:, 0].max()
        ymin = points[:, 1].min()
        ymax = points[:, 1].max()
        # Identify the four corners of the large rectangle in 3D
        large_rect_points = [
            (xmin, ymin),
            (xmin, ymax),
            (xmax, ymax),
            (xmax, ymin)
        ]

        # Remove these points from the original list
        small_rect_points = [point for point in points if tuple(point) not in large_rect_points]

        # Ensure we have exactly 4 points for each rectangle
        assert len(small_rect_points) == 4, "Small rectangle does not have exactly 4 points"
    
        # Function to sort points in the order: bottom-left, top-left, top-right, bottom-right
        def sort_rectangle(pts):
            pts = sorted(pts, key=lambda p: (p[0], p[1]))  # First sort by x, then by y
            return [pts[0], pts[1], pts[3], pts[2]]  # Rearrange to bottom-left, top-left, top-right, bottom-right

        # Sort points within each rectangle
        large_rect_points = sort_rectangle(large_rect_points)
        small_rect_points = sort_rectangle(small_rect_points)
        
        # Combine results into a single np.array in the specified order
        sorted_points = np.array([large_rect_points[0], large_rect_points[1], 
                                small_rect_points[0], small_rect_points[1], 
                                large_rect_points[2], small_rect_points[2], 
                                large_rect_points[3], small_rect_points[3]])
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

def create_thick_wall(stage, vertex, mesh_prim_path, transform_matrix, thcikness = 0.05):


    mesh = UsdGeom.Mesh.Define(stage, mesh_prim_path)

    constant_dim = np.where(np.all(vertex == vertex[0, :], axis=0))[0][0]

    back_vertex = np.copy(vertex)
    back_vertex[:, constant_dim] += thcikness

    # back_vertex = vertex + np.array([0, thcikness])

    points = convert_to_gf_vec3f_list(np.vstack((vertex, back_vertex)))

    mesh.GetPointsAttr().Set(points)
    if vertex.shape[0] == 8:
        faceVertexCounts = [3, 3, 3, 3, 3, 3, 3, 3,  #front with hole
                            3, 3, 3, 3, 3, 3, 3, 3,  #back with hole
                            3,3, #left outside
                            3,3, #right outside
                            3,3, #top outside
                            3,3, #bottom outside
                            3,3, #left inside
                            3,3, #right inside
                            3,3, #top inside
                            3,3 ]#bottom inside
        

        mesh.GetFaceVertexCountsAttr().Set(faceVertexCounts)
        faceVertexIndices = [1, 0, 2, 3, 1, 2, 4, 1, 3, 4, 3, 5, 6, 4, 5, 6, 5, 7, 6, 7, 0, 2, 0, 7,#front with hole
                            
                            8, 9, 10, 9, 11, 10, 9, 12, 11, 11, 12, 13, 12, 14, 13, 13, 14, 15, 15, 14, 8, 8, 10, 15,#back with hole

                            #  9, 8, 10, 11, 9, 10, 12, 9, 11, 12, 11, 13, 14, 12, 13, 14, 13, 15, 14, 15, 8, 10, 8, 15,#back with hole

                            1, 9, 8, 1, 8, 0, #left outside
                            4, 6, 14, 4, 14, 12, #right outside
                            1, 4, 12, 1, 12, 9, #top outside
                            0, 8, 14, 0, 14, 6, #bottom outside
                            
                            2, 10, 11, 2, 11, 3, #left inside
                            5, 13, 15, 5, 15, 7, #right inside
                            11, 13, 5, 11, 5, 3, #top inside
                            2, 7, 15, 2, 15, 10, #bottom inside
                            
                            ]
        mesh.GetFaceVertexIndicesAttr().Set(faceVertexIndices)

        # 设置法线
        normals = [
            Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), 
            Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), 
            Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), 
            Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), 
            Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), 
            Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), 
            Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), 
            Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), #front with hole

            Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1),
            Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1),
            Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1),
            Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1),
            Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1),
            Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1),
            Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1),
            Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1), #back with hole

            Gf.Vec3f(-1, 0, 0), Gf.Vec3f(-1, 0, 0), Gf.Vec3f(-1, 0, 0), 
            Gf.Vec3f(-1, 0, 0), Gf.Vec3f(-1, 0, 0), Gf.Vec3f(-1, 0, 0),#left outside

            Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 0, 0),
            Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 0, 0),#right outside

            Gf.Vec3f(0, 1, 0), Gf.Vec3f(0, 1, 0), Gf.Vec3f(0, 1, 0),
            Gf.Vec3f(0, 1, 0), Gf.Vec3f(0, 1, 0), Gf.Vec3f(0, 1, 0),#top outside

            Gf.Vec3f(0, -1, 0), Gf.Vec3f(0, -1, 0), Gf.Vec3f(0, -1, 0),
            Gf.Vec3f(0, -1, 0), Gf.Vec3f(0, -1, 0), Gf.Vec3f(0, -1, 0),#bottom outside

            Gf.Vec3f(0, 1, 0), Gf.Vec3f(0, 1, 0), Gf.Vec3f(0, 1, 0),
            Gf.Vec3f(0, 1, 0), Gf.Vec3f(0, 1, 0), Gf.Vec3f(0, 1, 0),#left inside

            Gf.Vec3f(0, -1, 0), Gf.Vec3f(0, -1, 0), Gf.Vec3f(0, -1, 0),
            Gf.Vec3f(0, -1, 0), Gf.Vec3f(0, -1, 0), Gf.Vec3f(0, -1, 0),#right inside

            Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1),
            Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1), Gf.Vec3f(0, 0, -1), #top inside

            Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1),
            Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1) #bottom inside

        ]


    elif vertex.shape[0] == 4:
        faceVertexCounts = [3, 3, #front
                            3, 3, #back
                            3, 3, #left
                            3, 3, #right
                            3, 3, #top
                            3, 3, #bottom
                              ]
        mesh.GetFaceVertexCountsAttr().Set(faceVertexCounts)


        faceVertexIndices = [0, 2, 1, 0, 3, 2,  #front
                            4, 5, 6, 4, 6, 7, #back
                            0, 1, 5, 0, 5, 4, #left
                            3, 6, 2, 3, 7, 6, #right
                            1, 2, 6, 1, 6, 5, #top
                            0, 4, 7, 0, 7, 3  #bottom
                            ]
        
        mesh.GetFaceVertexIndicesAttr().Set(faceVertexIndices)

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


    else:
        raise ValueError("Invalid number of vertices for a cuboid.")


    mesh.GetNormalsAttr().Set(normals)

    mesh.SetNormalsInterpolation("faceVarying")    #could be "vertex", "faceVarying", "uniform", "constant"

    material_global = UsdShade.Material.Get(stage, "/World/Materials/WhiteMarble__115104")
    UsdShade.MaterialBindingAPI(mesh).Bind(material_global)


    mesh.GetDoubleSidedAttr().Set(True)

    target_mesh_geom = UsdGeom.Xformable(mesh)
    target_mesh_geom.AddTransformOp().Set(transform_matrix)


def creat_thick_floor(stage, vertex, mesh_prim_path, transform_matrix, thcikness = 0.05):
    pass

if __name__ == "__main__":
    usd_file_path = "/home/zgao/house_usd_yup/train_5/house_train_5_processed.usda"
    base_path = "/World/Structure/Walls"
    new_file_path = "/home/zgao/house_usd_yup/train_5/wall_test.usda"

    process_structure_meshes(usd_file_path, base_path, new_file_path)