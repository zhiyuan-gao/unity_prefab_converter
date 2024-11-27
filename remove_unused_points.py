from pxr import Usd, UsdGeom, Sdf

def remove_unused_points(usda_file_path, output_file_path):
    """
    Remove unused points from a USD file and update all related attributes.
    
    Args:
        usda_file_path (str): Path to the input USDA file.
        output_file_path (str): Path to save the output USDA file.
    """
    # Open the USD stage
    stage = Usd.Stage.Open(usda_file_path)
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        
        mesh = UsdGeom.Mesh(prim)

        # Retrieve the original attributes
        points = mesh.GetPointsAttr().Get()
        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
        
        # Extract related Primvars
        primvars_api = UsdGeom.PrimvarsAPI(mesh)
        uv_primvar = primvars_api.GetPrimvar("UVMap")
        uv_values = uv_primvar.Get() if uv_primvar and uv_primvar.IsDefined() else None

        normals_attr = mesh.GetNormalsAttr()
        normals = normals_attr.Get() if normals_attr and normals_attr.IsDefined() else None

        sharp_face_attr = mesh.GetPrim().GetAttribute("primvars:sharp_face")
        sharp_face = sharp_face_attr.Get() if sharp_face_attr and sharp_face_attr.IsDefined() else None

        # Determine used indices
        used_indices = set(face_vertex_indices)
        new_index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(used_indices))}

        # Update points
        new_points = [points[idx] for idx in sorted(used_indices)]

        # Update faceVertexIndices
        new_face_vertex_indices = [new_index_map[idx] for idx in face_vertex_indices]

        # # Update UVMap
        # new_uv_values = None
        # if uv_values:
        #     if uv_primvar.GetInterpolation() == "vertex":
        #         new_uv_values = [uv_values[idx] for idx in sorted(used_indices)]
        #     elif uv_primvar.GetInterpolation() == "faceVarying":
        #         new_uv_values = [uv_values[idx] for idx in range(len(new_face_vertex_indices))]

        # # Update normals
        # new_normals = None
        # if normals:
        #     if normals_attr.GetMetadata("interpolation") == "vertex":
        #         new_normals = [normals[idx] for idx in sorted(used_indices)]
        #     elif normals_attr.GetMetadata("interpolation") == "faceVarying":
        #         new_normals = [normals[idx] for idx in range(len(new_face_vertex_indices))]

        # # Update sharp_face
        # new_sharp_face = None
        # if sharp_face:
        #     new_sharp_face = sharp_face[:len(face_vertex_counts)]  # Same length as face counts

        # Update attributes in the Mesh
        mesh.GetPointsAttr().Set(new_points)
        mesh.GetFaceVertexIndicesAttr().Set(new_face_vertex_indices)
        # if new_uv_values and uv_primvar:
        #     uv_primvar.Set(new_uv_values)
        # if new_normals:
        #     normals_attr.Set(new_normals)
        # if new_sharp_face is not None:
        #     sharp_face_attr.Set(new_sharp_face)

    # Save the updated USD file
    stage.GetRootLayer().Export(output_file_path)

# Example usage
input_usda = "/home/zgao/unity_preafab_converter/procthor_assets/Floor_Lamp_19/FloorLamp_19_GeomSubset_2.usda"
output_usda = "/home/zgao/unity_preafab_converter/procthor_assets/Floor_Lamp_19/FloorLamp_19_GeomSubset_2_simple.usda"
remove_unused_points(input_usda, output_usda)
