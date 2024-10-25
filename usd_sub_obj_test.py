from pxr import Usd, UsdGeom, UsdShade, Sdf, UsdUtils

# 打开原始 USD 文件
stage = Usd.Stage.Open('/home/zgao/ProcTHOR_Converter/procthor_assets_center_shifted/Floor_Lamp_19/FloorLamp_19.usda')

# Path to the parent mesh
mesh_path = '/root/FloorLamp_19/FloorLamp_19'

# 获取 Mesh Prim
mesh_prim = stage.GetPrimAtPath(mesh_path)
geom_mesh = UsdGeom.Mesh(mesh_prim)

# 获取该 mesh 的所有 GeomSubset
geom_subsets = UsdGeom.Subset.GetAllGeomSubsets(geom_mesh)

# 获取原始的顶点、法线、面顶点计数和面顶点索引
original_points = geom_mesh.GetPointsAttr().Get()
original_normals = geom_mesh.GetNormalsAttr().Get()
original_face_vertex_counts = geom_mesh.GetFaceVertexCountsAttr().Get()
original_face_vertex_indices = geom_mesh.GetFaceVertexIndicesAttr().Get()

# 遍历每个 GeomSubset
for i, subset in enumerate(geom_subsets):
    subset_name = subset.GetPrim().GetName()  # 获取 GeomSubset 的名字

    # 获取该 GeomSubset 的面索引
    indices = subset.GetIndicesAttr().Get()
    if not indices:
        print(f"Warning: No indices found for subset {subset_name}")
        continue

    # 创建一个新的 Prim，作为新的 Mesh
    new_prim_path = f'/GeomSubset_{i}_Prim'
    new_prim = stage.DefinePrim(new_prim_path, 'Mesh')
    new_geom_mesh = UsdGeom.Mesh(new_prim)

    # 用于存储子集的顶点、法线和面顶点索引
    subset_points = []
    subset_normals = []
    face_vertex_indices = []
    subset_face_vertex_counts = []

    # 用于追踪原顶点的索引映射
    index_mapping = {}
    new_index = 0

    # 遍历每个 `GeomSubset` 中的面索引 `indices`
    face_start_index = 0  # 当前面在 faceVertexIndices 中的起始索引
    for face_id in indices:
        # 获取该面的顶点数量
        vertex_count = original_face_vertex_counts[face_id]
        subset_face_vertex_counts.append(vertex_count)

        # 获取该面的顶点索引
        face_indices = original_face_vertex_indices[face_start_index:face_start_index + vertex_count]

        # 映射并添加顶点和法线数据
        for index in face_indices:
            if index not in index_mapping:
                index_mapping[index] = new_index
                subset_points.append(original_points[index])
                subset_normals.append(original_normals[index])
                new_index += 1

            face_vertex_indices.append(index_mapping[index])

        # 更新起始索引到下一个面的位置
        face_start_index += vertex_count

    # 创建新的顶点和法线属性
    new_geom_mesh.CreatePointsAttr(subset_points)
    new_geom_mesh.CreateNormalsAttr(subset_normals)

    # 设置面顶点索引和面顶点计数
    new_geom_mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    new_geom_mesh.CreateFaceVertexCountsAttr(subset_face_vertex_counts)

    # 绑定对应的材质
    material_prim_path = f'/_materials/{subset_name}'
    material_prim = stage.GetPrimAtPath(material_prim_path)
    if material_prim:
        material_binding_api = UsdShade.MaterialBindingAPI(new_geom_mesh)
        material_binding_api.Bind(UsdShade.Material(material_prim))

# 保存为新的 USD 文件
new_usd_path = '/home/zgao/ProcTHOR_Converter/procthor_assets_center_shifted/Floor_Lamp_19/new_file.usda'
stage.GetRootLayer().Export(new_usd_path)

print(f"New USD file saved as {new_usd_path}")
