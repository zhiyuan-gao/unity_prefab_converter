from pxr import Usd, UsdGeom, UsdShade, Sdf, UsdUtils
import os
import re

def split_usd(usd_file_path):
    
    stage = Usd.Stage.Open(usd_file_path)
    usd_file_name = os.path.splitext(usd_file_path)[0]
    usd_file_local_name = os.path.splitext(os.path.basename(usd_file_path))[0]
    out_put_path_list = []
    default_prim = stage.GetDefaultPrim()
    xform_prim = [prim for prim in default_prim.GetChildren() if UsdGeom.Xform(prim)][0]
    
    mesh_prim = xform_prim.GetChildren()[0]

    geom_mesh = UsdGeom.Mesh(mesh_prim)
    # 获取该 mesh 的所有 GeomSubset
    geom_subsets = UsdGeom.Subset.GetAllGeomSubsets(geom_mesh)

    # 获取原始的顶点、法线、面顶点计数和面顶点索引
    original_points = geom_mesh.GetPointsAttr().Get()
    original_normals = geom_mesh.GetNormalsAttr().Get()
    original_face_vertex_counts = geom_mesh.GetFaceVertexCountsAttr().Get()
    original_face_vertex_indices = geom_mesh.GetFaceVertexIndicesAttr().Get()
    # print('original_face_vertex_indices',original_face_vertex_indices)
    face_start_index = 0  # 当前面在 faceVertexIndices 中的起始索引

    # 遍历每个 GeomSubset
    for i, subset in enumerate(geom_subsets):
        subset_name = subset.GetPrim().GetName()  # 获取 GeomSubset 的名字

        # 获取该 GeomSubset 的面索引
        indices = subset.GetIndicesAttr().Get()
        if not indices:
            print(f"Warning: No indices found for subset {subset_name}")
            continue

        # 创建一个新的 Prim，作为新的 Mesh
        new_prim_path = f'/{usd_file_local_name}_GeomSubset_{i}'
        new_prim = stage.DefinePrim(new_prim_path, 'Mesh')
        new_geom_mesh = UsdGeom.Mesh(new_prim)

        # 用于存储子集的顶点、法线和面顶点索引
        subset_points = []
        subset_normals = []
        geom_vertex_indices = []
        subset_face_vertex_counts = []

        # 用于追踪原顶点的索引映射
        index_mapping = {}
        new_index = 0

        # 遍历每个 `GeomSubset` 中的面索引 `indices`

        for face_id in indices:
            # 获取该面的顶点数量
            vertex_count = original_face_vertex_counts[face_id]
            subset_face_vertex_counts.append(vertex_count)
            face_indices = original_face_vertex_indices[face_start_index:face_start_index + vertex_count]
            geom_vertex_indices.extend(face_indices)
            # 更新起始索引到下一个面的位置
            face_start_index += vertex_count

        # 创建新的顶点和法线属性
        subset_points = original_points
        subset_normals = original_normals

        new_geom_mesh.CreatePointsAttr(subset_points)
        new_geom_mesh.CreateNormalsAttr(subset_normals)

        # 设置面顶点索引和面顶点计数
        new_geom_mesh.CreateFaceVertexIndicesAttr(geom_vertex_indices)
        # print('geom_vertex_indices',geom_vertex_indices)
        new_geom_mesh.CreateFaceVertexCountsAttr(subset_face_vertex_counts)

        # 绑定对应的材质
        material_prim_path = f'/_materials/{subset_name}'
        material_prim = stage.GetPrimAtPath(material_prim_path)
        if material_prim:
            material_binding_api = UsdShade.MaterialBindingAPI(new_geom_mesh)
            material_binding_api.Bind(UsdShade.Material(material_prim))
        
        # output_usd_path = os.path.join(usd_file_name, f"{subset_name}.usda")
        output_usd_path = f'{usd_file_name}_GeomSubset_{i}.usda'
        out_put_path_list.append(output_usd_path)
        # output_usd_path = os.path.join(source_dir, 'procthor_assets', asset_id, f"{subset_name}.usda")

        # 创建一个新的 USD 文件 Stage
        new_stage = Usd.Stage.CreateNew(output_usd_path)
        new_stage.SetMetadata("upAxis", "Z")
        
        # 使用 Sdf.CopySpec 复制 new_prim 到新文件的根层
        Sdf.CopySpec(stage.GetRootLayer(), new_prim.GetPath(), new_stage.GetRootLayer(), new_prim.GetPath())
        
        # 保存新文件
        new_stage.GetRootLayer().Save()
    
    return out_put_path_list


if __name__=="__main__":


    source_dir = os.path.dirname(os.path.realpath(__file__))
    assets_output_dir = os.path.join(source_dir, 'procthor_assets')

    usd_file_path = '/home/zgao/unity_preafab_converter/procthor_assets/Floor_Lamp_19/FloorLamp_19.usda'
    # mesh_path = '/root/FloorLamp_19/FloorLamp_19'
    out_put_path_list = split_usd(usd_file_path)
    print(out_put_path_list)

