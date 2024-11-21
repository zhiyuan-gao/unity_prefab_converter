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

    # get UV data
    uv_attr = mesh_prim.GetAttribute("primvars:UVMap")
    uv_values = None

    if uv_attr and uv_attr.IsDefined():
        uv_values = uv_attr.Get()  
        interpolation = uv_attr.GetMetadata("interpolation")
 
    # get original geometry data, including points, normals, face vertex counts and face vertex indices
    original_points = geom_mesh.GetPointsAttr().Get()
    original_normals = geom_mesh.GetNormalsAttr().Get()
    normal_interpolation = geom_mesh.GetNormalsAttr().GetMetadata("interpolation")


    original_face_vertex_counts = geom_mesh.GetFaceVertexCountsAttr().Get()
    original_face_vertex_indices = geom_mesh.GetFaceVertexIndicesAttr().Get()

    # 使用 PrimvarsAPI 获取 primvars:sharp_face
    primvars_api = UsdGeom.PrimvarsAPI(geom_mesh)
    sharp_face_attr = primvars_api.GetPrimvar("sharp_face")

    if sharp_face_attr and sharp_face_attr.IsDefined():
        sharp_face_values = sharp_face_attr.Get()
       
    double_sided_attr = geom_mesh.GetDoubleSidedAttr()
    if double_sided_attr and double_sided_attr.IsDefined():
        double_sided_value = double_sided_attr.Get()


    ori_vertex_indices = []
    start_index = 0

    for i, vertex_count in enumerate(original_face_vertex_counts):
        ori_vertex_indices.append(original_face_vertex_indices[start_index:start_index + vertex_count])
        start_index += vertex_count

    # get GeomSubsets
    geom_subsets = UsdGeom.Subset.GetAllGeomSubsets(geom_mesh)

    # iterate over each GeomSubset
    for i, subset in enumerate(geom_subsets):

        # create a new stage for each GeomSubset
        output_usd_path = f'{usd_file_name}_GeomSubset_{i}.usda'
        new_stage = Usd.Stage.CreateNew(output_usd_path)
        new_stage.SetMetadata("upAxis", "Z")
        
        # set default prim
        root_prim = new_stage.DefinePrim("/root", "Xform")
        new_stage.SetDefaultPrim(root_prim)

        subset_name = subset.GetPrim().GetName()
        indices = subset.GetIndicesAttr().Get()
        if not indices:
            print(f"Warning: No indices found for subset {subset_name}")
            continue

        new_xform_prim_path = f'/root/{usd_file_local_name}_GeomSubset_{i}'
        new_xform_prim = new_stage.DefinePrim(new_xform_prim_path, 'Xform')

        new_mesh_prim_path = f'{new_xform_prim_path}/{usd_file_local_name}_GeomSubset_{i}'

        # create a new mesh prim for the geometry subset
        new_mesh_prim = new_stage.DefinePrim(new_mesh_prim_path, 'Mesh')
        new_geom_mesh = UsdGeom.Mesh(new_mesh_prim)

        subset_face_vertex_counts = []
        geom_vertex_indices = []

        for face_id in indices:
            vertex_count = original_face_vertex_counts[face_id]
            subset_face_vertex_counts.append(vertex_count)
            face_indices = ori_vertex_indices[face_id]
            geom_vertex_indices.extend(face_indices)

  
        # set new geometry data
        new_geom_mesh.CreatePointsAttr(original_points)
        new_geom_mesh.CreateFaceVertexIndicesAttr(geom_vertex_indices)
        new_geom_mesh.CreateFaceVertexCountsAttr(subset_face_vertex_counts)

        # set new normal attribute
        normals_attr = new_geom_mesh.CreateNormalsAttr()

        new_normals = [original_normals[vertex_id] for vertex_id in geom_vertex_indices]
        normals_attr.Set(new_normals)  
        normals_attr.SetMetadata("interpolation", normal_interpolation)

        # creat PrimvarsAPI for new mesh
        primvars_api = UsdGeom.PrimvarsAPI(new_geom_mesh)

        # set sharp_face primvar
        if sharp_face_values:
            subset_sharp_face_values = [sharp_face_values[face_id] for face_id in indices]
            sharp_face_primvar = primvars_api.CreatePrimvar(
                "sharp_face", Sdf.ValueTypeNames.BoolArray, UsdGeom.Tokens.uniform
            )
            sharp_face_primvar.Set(subset_sharp_face_values)

        new_geom_mesh.GetDoubleSidedAttr().Set(double_sided_value)


        # set new UV data
        if uv_values:
            sub_uv_values = [uv_values[vertex_id] for vertex_id in geom_vertex_indices]
            uv_primvar = primvars_api.CreatePrimvar("UVMap", Sdf.ValueTypeNames.TexCoord2fArray, interpolation)
            uv_primvar.Set(sub_uv_values)

        # get material binding
        subset_material_binding_api = UsdShade.MaterialBindingAPI(subset.GetPrim())
        bound_material = subset_material_binding_api.GetDirectBindingRel().GetTargets()

        materials_scope_path = "/root/_materials"
        materials_scope = new_stage.DefinePrim(materials_scope_path, "Scope")

        if bound_material:
            for material_path in bound_material:
                material_prim = stage.GetPrimAtPath(material_path)
                if material_prim:

                    new_material_path = f"{materials_scope_path}/{material_prim.GetName()}"
                    Sdf.CopySpec(stage.GetRootLayer(), material_prim.GetPath(), new_stage.GetRootLayer(), new_material_path)

                    new_material_binding_api = UsdShade.MaterialBindingAPI(new_geom_mesh)
                    new_material_binding_api.Bind(UsdShade.Material(new_stage.GetPrimAtPath(new_material_path)))

        # save the new usda
        new_stage.GetRootLayer().Save()
        out_put_path_list.append(output_usd_path)

    return out_put_path_list

if __name__=="__main__":


    source_dir = os.path.dirname(os.path.realpath(__file__))
    assets_output_dir = os.path.join(source_dir, 'procthor_assets')

    usd_file_path = '/home/zgao/unity_preafab_converter/procthor_assets/Floor_Lamp_19/FloorLamp_19.usda'
    # mesh_path = '/root/FloorLamp_19/FloorLamp_19'
    out_put_path_list = split_usd(usd_file_path)
    # print(out_put_path_list)

