#!/usr/bin/env python3.11

import bpy
import re
import os
# from mathutils import Vector, Euler
from math import radians
from mathutils import Matrix, Quaternion, Vector
import yaml
import math
import shutil
import subprocess
import json

source_dir = os.path.dirname(os.path.realpath(__file__))
assets_output_dir = os.path.join(source_dir, 'procthor_assets')

def find_asset_path_by_guid(unity_asset_path, guid):
    """
    Find the asset path by searching for a GUID in .meta files.
    
    :param unity_asset_path: The root directory to search for the asset.
    :param guid: The GUID to search for.
    :return: The path to the asset (without .meta) if found, otherwise None.
    """
    if guid:
            
        for root, dirs, files in os.walk(unity_asset_path):
            for file in files:
                if file.endswith('.meta'):
                    meta_path = os.path.join(root, file)
                    with open(meta_path, 'r') as meta_file:
                        if guid in meta_file.read():
                            return meta_path[:-5]  # remove ".meta"

        # If not found, log missing GUID and print a warning
        print(f"Warning: Resource with GUID {guid} not found.")
        return None
    
    # If guid is None, return None
    else:
        return None

def parse_unity_mat_file(filepath):
    with open(filepath, 'r') as file:
        # jump over YAML document tags
        content = file.readlines()
        content = [line for line in content if not line.startswith(('%YAML', '%TAG', '--- !u!'))]
        mat_content = yaml.safe_load("".join(content))
    
    material_info = {}

    # extract material name
    material_info['Name'] = mat_content.get('Material', {}).get('m_Name', 'UnnamedMaterial')

    # extract shader keywords
    material_info['ShaderKeywords'] = mat_content.get('Material', {}).get('m_ShaderKeywords', '')

    # extract texture information
    material_info['Textures'] = []
    saved_properties = mat_content.get('Material', {}).get('m_SavedProperties', {})
    tex_envs = saved_properties.get('m_TexEnvs', [])

    for tex_env in tex_envs:
        texture_info = {}
        texture_name = tex_env['first']['name']
        texture_info['name'] = texture_name
        
        # get texture path or GUID
        texture_guid = tex_env['second']['m_Texture'].get('guid', '')
        texture_info['textureGUID'] = texture_guid

        # get scaling and offset
        texture_info['scale'] = tex_env['second'].get('m_Scale', {'x': 1, 'y': 1})
        texture_info['offset'] = tex_env['second'].get('m_Offset', {'x': 0, 'y': 0})

        material_info['Textures'].append(texture_info)

    return material_info


def snake_to_camel(snake_str):
     # Split the string by underscores, but keep numbers separated
    components = re.split(r'(_\d+_)|(_\d+)|(_\d+)|_', snake_str)
    
    # Filter out None or empty strings that might result from the split
    components = [comp for comp in components if comp]
    
    # Capitalize the first letter of each component except those that are purely numeric
    camel_str = ''.join(comp.title() if not comp.isdigit() else comp for comp in components)

    camel_str = re.sub(r'(\d+)', r'_\1_', camel_str)

    camel_str = camel_str.strip('_').replace('.', '_').replace(' ', '_').replace('__', '_').replace(':', '_')
    
    return camel_str

def select_object_with_children(obj, reset_origin=True):
    # Select the parent object
    obj.select_set(True)

    # Set the origin to geometry
    if reset_origin:
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    # Recursively select all children
    for child in obj.children:
        select_object_with_children(child)

def convert_prefab(asset_id, prefab_info, root_path='/home/zgao/ai2thor/unity', shift_center=False):

# if asset_id in all_prefab_details:
    # prefab_info = all_prefab_details[asset_id]
    asset_dir = os.path.join(assets_output_dir, asset_id)

    if not os.path.exists(asset_dir):
        os.makedirs(asset_dir)

    for sub_obj in prefab_info:
        data = bpy.data
        for armature in data.armatures:
            data.armatures.remove(armature)
        for mesh in data.meshes:
            data.meshes.remove(mesh)
        for object in data.objects:
            data.objects.remove(object)
        for material in data.materials:
            data.materials.remove(material)
        for camera in data.cameras:
            data.cameras.remove(camera)
        for light in data.lights:
            data.lights.remove(light)
        for image in data.images:
            data.images.remove(image)

        mesh_path = os.path.join(root_path, prefab_info[sub_obj]['MeshPath'])

        mesh_name = prefab_info[sub_obj]['MeshName']
        # print('mesh_name:', mesh_name)

        mesh_file_extension = os.path.splitext(mesh_path)[1]
        if mesh_file_extension == ".fbx":
            bpy.ops.import_scene.fbx(filepath=mesh_path, bake_space_transform = True)
            # Deselect all objects first
            bpy.ops.object.select_all(action='DESELECT')

            for _obj in bpy.context.scene.objects:
                if _obj.name != mesh_name:
                    _obj.select_set(True)
                else:
                    _obj.select_set(False)
            # delete the other objects
            bpy.ops.object.delete()

            obj = bpy.data.objects.get(mesh_name)

        elif mesh_file_extension == ".obj":
            bpy.ops.wm.obj_import(filepath=mesh_path, up_axis='Z', forward_axis='Y')
            # Deselect all objects first
            bpy.ops.object.select_all(action='DESELECT')
            # only one object in the .obj file
            obj = bpy.context.scene.objects[0]
            # the mesh_name in obj file is 'default', use asset_id instead
            obj.name = asset_id

        else:
            print(f"Unsupported mesh file extension {mesh_file_extension}")
            continue

        print('processing:', obj.name)
        # Find and select the specified Mesh object

        if obj:

            obj.hide_set(False) 
            obj.hide_viewport = False  
            obj.select_set(True)
            for child in obj.children:
                child.select_set(False)

            bpy.context.view_layer.objects.active = obj

            position_unity = prefab_info[sub_obj]['Transform']["Position"]
            position_blender = Vector((
                -position_unity["x"],  
                -position_unity["z"],  
                position_unity["y"]   
            )) 

            rotation_unity = prefab_info[sub_obj]['Transform']["Rotation"]

            rotation_blender = convert_unity_rotation_to_blender(rotation_unity)

            scale_unity = prefab_info[sub_obj]['Transform']["Scale"]
            scale_blender = Vector((
                scale_unity["x"],      
                scale_unity["z"],      
                scale_unity["y"] 
            ))

            transform_matrix_pivot = (
                Matrix.Translation(position_blender) @
                rotation_blender.to_matrix().to_4x4() @
                Matrix.Diagonal(scale_blender).to_4x4()
            )
            # print('position_blender',position_blender)
            # print('rotation_blender',rotation_blender)
            # print('scale_blender',scale_blender)            

            # if obj.location == (0, 0, 0): 
            # if obj.matrix_world.translation  == (0, 0, 0):
            epsilon = 1e-6
            global_location = obj.matrix_world.translation
            # consider this as no pivot found
            if abs(global_location.x) < epsilon and abs(global_location.y) < epsilon and abs(global_location.z) < epsilon:

                # _____________________set the correct pivot of fbx ______________________
        
                original_transform = prefab_info[sub_obj].get('OriginalTransform')
                # ['OriginalTransform']
                if original_transform is not None:


                    original_translate = Vector((-original_transform['Position']['x'],
                                                -original_transform['Position']['z'],
                                                original_transform['Position']['y']))

                    original_rotate = convert_unity_rotation_to_blender(original_transform['Rotation'])

                    original_scale = Vector((original_transform['Scale']['x'],
                                                original_transform['Scale']['z'],
                                                original_transform['Scale']['y']))
                    
                else:
                    # if the original transform is not provided, it is the single mesh in the fbx and prefab.
                    original_translate = position_blender
                    original_rotate = rotation_blender
                    original_scale = scale_blender
                                                

                inverse_rotate = original_rotate.conjugated() 

                inverse_rotation_matrix = inverse_rotate.to_matrix().to_4x4()  

                inverse_scale_matrix = Matrix.Diagonal(Vector((1.0 / original_scale.x, 
                                                            1.0 / original_scale.y, 
                                                            1.0 / original_scale.z))).to_4x4()

                inverse_translate = -original_translate
                inverse_translation_matrix = Matrix.Translation(inverse_translate)

                inverse_transform_matrix = inverse_translation_matrix @ inverse_rotation_matrix @ inverse_scale_matrix

                obj.matrix_world = inverse_transform_matrix  @ obj.matrix_world

                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

                # ____________________adapt the pivot to prefab ______________________

                obj.matrix_world =  transform_matrix_pivot @ obj.matrix_world
                # obj.matrix_world =  transform_matrix_pivot 
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            else:
                # already has a pivot
                obj.matrix_world =  transform_matrix_pivot
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            obj.name = snake_to_camel(obj.name)
            print('obj.name:', obj.name)
            obj.data.name = f"SM_{obj.name}"

            # Prefab's pivot is the center of the all sub-meshes, got from unity
            if shift_center:
                box_center = Vector((
                    -prefab_info[sub_obj]['BoxCenter']["x"],      
                    -prefab_info[sub_obj]['BoxCenter']["z"],      
                    prefab_info[sub_obj]['BoxCenter']["y"] 
                ))

                transfrom_box_center = Matrix.Translation(-box_center)
                obj.matrix_world = transfrom_box_center @ obj.matrix_world
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            if not obj.data.materials:
                obj.data.materials.append(None)  
            for index, material in enumerate(obj.data.materials):
                if material:  # Check if the material slot is not empty
                    # Rename the material
                    material.name = f"M_{obj.name}_{index}"

                else:  
                    new_material = bpy.data.materials.new(name=f"M_{obj.name}_{index}")
                    obj.data.materials[index] = new_material
                        
            material_slots = prefab_info[sub_obj]['Materials']
            # if more than one material, split the mesh


            # Define the output file path
            obj_file_path = os.path.join(asset_dir, f"{obj.name}.obj")
            
            # Export the selected object to an .obj file
            # blender 4.x
            bpy.ops.wm.obj_export(filepath=obj_file_path, export_selected_objects=True, forward_axis='Y', up_axis='Z')
            # blender 2.x
            # bpy.ops.export_scene.obj(filepath=obj_file_path, use_selection=True, axis_forward='Y', axis_up='Z')

            stl_file_path = os.path.join(asset_dir, f"{obj.name}.stl")
            # Export the selected object to an .stl file
            bpy.ops.wm.stl_export(filepath=stl_file_path, export_selected_objects=True, forward_axis='Y', up_axis='Z')# blender 4.x
            # bpy.ops.export_mesh.stl(filepath=stl_file_path, use_selection=True, axis_forward='Y', axis_up='Z')# blender 2.x
            # bpy.ops.wm.quit_blender()
            return obj_file_path

        
    else:
        print(f"Mesh {mesh_name} not found in {mesh_path}")
        return None


def convert_usd_to_obj(usda_file_path):

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=usda_file_path)
    obj = bpy.context.selected_objects[0]  
    if obj.type == 'MESH':
        # mat = bpy.data.materials.new(name=f'M_{{obj.name}}_{i}')
        mat = bpy.data.materials.new(name=f'M_{obj.name}')
        obj.data.materials.append(mat)

    usd_file_name = os.path.splitext(usda_file_path)[0]
    obj_path = f'{usd_file_name}.obj'
    bpy.ops.wm.obj_export(filepath=obj_path, export_selected_objects=True, forward_axis='Y', up_axis='Z')
    # print("OBJ file exported successfully:", obj_path)

def convert_obj_to_usd(obj_file_path,usda_file_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)

    bpy.ops.wm.obj_import(filepath=obj_file_path, up_axis='Z', forward_axis='Y')

    bpy.ops.wm.usd_export(filepath=usda_file_path, selected_objects_only=False)

def process_pipeline(asset_id, prefab_info, root_path='/home/zgao/ai2thor/unity',shift_center=False):


    obj_file_path = convert_prefab(asset_id, prefab_info, root_path, shift_center)

    add_texture(asset_id,prefab_info,root_path)

    obj_file_name = os.path.splitext(obj_file_path)[0]
    usda_file_path = f'{obj_file_name}.usda'

    convert_obj_to_usd(obj_file_path,usda_file_path)
    convert_usd_to_obj(usda_file_path)

    # split usd
    code = f"""
from usd_sub_obj_test import split_usd
import json
result = split_usd('{usda_file_path}')
print(json.dumps(result))
    """
    python_path = "/home/zgao/.virtualenvs/multiverse/bin/python"
    result = subprocess.run([python_path, "-c", code], capture_output=True, text=True)

    if result.returncode == 0:

        try:
            out_put_path_list = json.loads(result.stdout)

        except json.JSONDecodeError:
            print("Error: cannot decode the output")
    else:
        print("Error:", result.stderr)

    # convert the splited usd file to obj file and export to same folder
    print('hi')
    for i, sub_usda_file_path in enumerate(out_put_path_list):
        convert_usd_to_obj(sub_usda_file_path)

    

#             usda_file_path = os.path.join(asset_dir, f"{obj.name}.usda")

#             def convert_obj_to_usd(obj_file_path,usda_file_path):
#                 bpy.ops.wm.read_factory_settings(use_empty=True)
#                 bpy.ops.wm.obj_import(filepath=obj_file_path, up_axis='Z', forward_axis='Y')

#                 bpy.ops.wm.usd_export(filepath=usda_file_path, selected_objects_only=False)
#                 bpy.ops.wm.quit_blender()

#             convert_obj_to_usd(obj_file_path,usda_file_path)
#             if len(material_slots) > 1:

#                 # change the mtl.
                
#                 # usda_file_path = os.path.join(asset_dir, f"{obj.name}.usda")

#                 # convert obj to usd file
# #                 code = f"""
# # import bpy
# import os

# def convert_obj_to_usd(obj_file_path,usda_file_path):
#     bpy.ops.wm.read_factory_settings(use_empty=True)

#     bpy.ops.import_scene.obj(filepath=obj_file_path)

#     bpy.ops.wm.usd_export(filepath=usda_file_path, selected_objects_only=False)

# convert_obj_to_usd(r"{obj_file_path}")
#                 """
#                 process = subprocess.Popen(
#                     ["blender", "--background", "--python-expr", code],
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.PIPE,
#                     text=True,
#                     bufsize=1 
#                 )
#                 process.wait()

                # # export the complete mesh to usd file
                # usda_file_path = os.path.join(asset_dir, f"{obj.name}.usda")
                # bpy.ops.wm.usd_export(
                #     filepath=usda_file_path,
                #     selected_objects_only=True, 
#                 #     export_animation=False,       
#                 #     export_materials=True 
#                 # )

#                 # python executable path
#                 python_path = "/home/zgao/.virtualenvs/multiverse/bin/python"

#                 # split the usd file
#                 code = f"""
# from usd_sub_obj_test import split_usd
# import json
# result = split_usd('{usda_file_path}')
# print(json.dumps(result))
#                 """
#                 result = subprocess.run([python_path, "-c", code], capture_output=True, text=True)

#                 if result.returncode == 0:

#                     try:
#                         out_put_path_list = json.loads(result.stdout)

#                     except json.JSONDecodeError:
#                         print("Error: cannot decode the output")
#                 else:
#                     print("Error:", result.stderr)

#                 # convert the splited usd file to obj file and export to same folder
#                 for i, sub_usda_file_path in enumerate(out_put_path_list):
#                     # material.name = f"M_{obj.name}_{i}"
#                     code = f"""
# import bpy
# import os

# def convert_usd_to_obj(usda_file_path):

#     bpy.ops.wm.read_factory_settings(use_empty=True)
#     bpy.ops.wm.usd_import(filepath=usda_file_path)
#     obj = bpy.context.selected_objects[0]  
#     if obj.type == 'MESH':
#         # mat = bpy.data.materials.new(name=f'M_{{obj.name}}_{i}')
#         mat = bpy.data.materials.new(name=f'M_{{obj.name}}')
#         obj.data.materials.append(mat)

#     usd_file_name = os.path.splitext(usda_file_path)[0]
#     obj_path = f'{{usd_file_name}}.obj'
#     bpy.ops.wm.obj_export(filepath=obj_path, export_selected_objects=True, forward_axis='Y', up_axis='Z')
#     print("OBJ file exported successfully:", obj_path)

# convert_usd_to_obj(r"{sub_usda_file_path}")
#                     """

#                     process = subprocess.Popen(
#                         ["blender", "--background", "--python-expr", code],
#                         stdout=subprocess.PIPE,
#                         stderr=subprocess.PIPE,
#                         text=True,
#                         bufsize=1 
#                     )
#                     process.wait()





            # confer the splited usd file to obj file



    # else:
    #     print(f"Prefab {asset_id} not found in AllPrefabDetails.json")

def convert_unity_rotation_to_blender(unity_quat):
    """
  
    use mathutils to convert Unity's rotation quaternion to the target coordinate system (Z axis up, Y axis forward, X axis right)
    
    Input:
    unity_quat: dict, Unity's quaternion, containing 'x', 'y', 'z', 'w'

    Output:
    Quaternion, (w, x, y, z) in Blender's coordinate system
    """

    original_rotation = Quaternion([
        unity_quat['w'],  
        unity_quat['x'], 
        unity_quat['y'], 
        unity_quat['z']
    ])
    

    transform_matrix = Matrix([
        [1, 0, 0],  
        [0, 0, 1],  
        [0, 1, 0] 
    ])
    

    new_rotation_matrix = transform_matrix @ original_rotation.to_matrix() @ transform_matrix.transposed()

    new_rotation = new_rotation_matrix.to_quaternion()
    
    return new_rotation



def load_mat_file(filepath):
    with open(filepath, 'r') as file:
        # Remove unnecessary lines from the .mat file
        content = file.readlines()
        content = [line for line in content if not line.startswith(('%YAML', '%TAG', '--- !u!'))]
        mat_content = yaml.safe_load("".join(content))
    # mat_data = mat_content['Material']
    return mat_content

def modify_mtl_file(material_name, mat_data, texture_folder, root_path):
    """
    Modify an .mtl file based on the parsed .mat data and update texture paths to absolute paths.
    
    :param mtl_filepath: Path to the .mtl file.
    :param mat_data: Dictionary containing material properties from the .mat file.
    :param root_path: Root path to locate textures.
    """
    # with open(mtl_filepath, 'r') as file:
    #     mtl_lines = file.readlines()


    # Parse material name and properties from .mat data using parse_material_properties function
    # material_name = mat_data.get('Material', {}).get('m_Name', 'Material')
    colors, tex_envs, floats = parse_material_properties(mat_data)

    diffuse = colors.get("_Color", {'r': 0.8, 'g': 0.8, 'b': 0.8, 'a': 1})
    transparency = diffuse['a']

    ambient = colors.get("_SColor", {'r': 0.2, 'g': 0.2, 'b': 0.2, 'a': 1})
    emissive = colors.get("_EmissionColor", {'r': 0., 'g': 0., 'b': 0., 'a': 1})
    specular = colors.get("_SpecularColor", {'r': 0.2, 'g': 0.2, 'b': 0.2, 'a': 1})
    shininess = floats.get("_Shininess", 20.0)
    Metalness = floats.get("_Metallic", 0.0)

    # reflection = colors.get("_ReflectColor",  {'r': 0., 'g': 0., 'b': 0., 'a': 1})
    # _Glossiness, _ReflectColor have no direct equivalent in .mtl files

    # Find the texture paths and convert them to absolute paths
    main_tex_guid = tex_envs.get('_MainTex', {}).get('m_Texture', {}).get('guid', '')
    main_tex_path = find_asset_path_by_guid(root_path, main_tex_guid)
    main_tex_path_abs = os.path.abspath(main_tex_path) if main_tex_path else None

    bump_map_guid = tex_envs.get('_BumpMap', {}).get('m_Texture', {}).get('guid', '')
    bump_map_path = find_asset_path_by_guid(root_path, bump_map_guid)
    bump_map_path_abs = os.path.abspath(bump_map_path) if bump_map_path else None

    specular_map_guid = tex_envs.get('_SpecGlossMap', {}).get('m_Texture', {}).get('guid', '')
    specular_map_path = find_asset_path_by_guid(root_path, specular_map_guid)
    specular_map_path_abs = os.path.abspath(specular_map_path) if specular_map_path else None

    sb = []
    sb.append(f"newmtl {material_name}")
    # Ka r g b
    # defines the ambient color of the material to be (r,g,b). The default is (0.2,0.2,0.2);
    sb.append(f"Ka {ambient['r']:.8f} {ambient['g']:.8f} {ambient['b']:.8f}")

    # Kd r g b
    # defines the diffuse color of the material to be (r,g,b). The default is (0.8,0.8,0.8);
    sb.append(f"Kd {diffuse['r']:.8f} {diffuse['g']:.8f} {diffuse['b']:.8f}")

    # Ks r g b
    # defines the specular color of the material to be (r,g,b). This color shows up in highlights. The default is (1.0,1.0,1.0);
    sb.append(f"Ks {specular['r']:.8f} {specular['g']:.8f} {specular['b']:.8f}")

    # Tf r g b
    # defines the transmission filter of the material to be (r,g,b). This is the color that is multiplied by Kd to get the actual color that is displayed. The default is (1.0,1.0,1.0);
    
    # Ke r g b
    # defines the emissive color of the material to be (r,g,b). This is the color of the material when it is self-luminous. The default is (0.0,0.0,0.0);
    sb.append(f"Ke {emissive['r']:.8f} {emissive['g']:.8f} {emissive['b']:.8f}")
    # Ns s
    # defines the shininess of the material to be s. The default is 0.0;
    sb.append(f"Ns {shininess:.8f}")

    # Tr alpha
    # defines the transparency of the material to be alpha. The default is 0.0 (not transparent at all). The quantities d and Tr are the opposites of each other, and specifying transparency or nontransparency is simply a matter of user convenience.
    sb.append(f"Tr {transparency:.8f}")
    # d alpha, d and tr are opposites, one is enough
    # defines the non-transparency of the material to be alpha. The default is 1.0 (not transparent at all). The quantities d and Tr are the opposites of each other, and specifying transparency or nontransparency is simply a matter of user convenience.
    # sb.append(f"d {1 - transparency:.8f}")

    # Pm s
    # defines the metallic reflectivity of the material to be s. The default is 0.0;
    sb.append(f"Pm {Metalness:.8f}")

    # illum n
    # denotes the illumination model used by the material. illum = 1 indicates a flat material with no specular highlights, so the value of Ks is not used. illum = 2 denotes the presence of specular highlights, and so a specification for Ks is required.
    sb.append("illum 2")
    

    # map_Ka filename
    # names a file containing a texture map, which should just be an ASCII dump of RGB values;
    if main_tex_path_abs:
        main_tex_path_rel = copy_file_and_get_relative_path(main_tex_path_abs, texture_folder)
        sb.append(f"map_Kd {main_tex_path_rel}")
        # sb.append(f"map_Kd {main_tex_path_abs}")

    if bump_map_path_abs:
        bump_map_path_rel = copy_file_and_get_relative_path(bump_map_path_abs, texture_folder)
        sb.append(f"map_Bump {bump_map_path_rel}")
        # sb.append(f"map_Bump {bump_map_path_abs}")

    if specular_map_path_abs:
        specular_map_path_rel = copy_file_and_get_relative_path(specular_map_path_abs, texture_folder)
        sb.append(f"map_Ks {specular_map_path_rel}")
        # sb.append(f"map_Ks {specular_map_path_abs}")

    ret = "\n".join(sb)

    return ret


def parse_material_properties(mat_data):
    """
    Parse material properties from .mat data considering both formats.
    
    :param mat_data: Dictionary containing material properties.
    :return: Parsed color, texture, and float dictionaries.
    """
    saved_properties = mat_data.get('Material', {}).get('m_SavedProperties', {})
    colors_raw = saved_properties.get('m_Colors', [])
    tex_envs_raw = saved_properties.get('m_TexEnvs', [])
    floats_raw = saved_properties.get('m_Floats', [])

    colors = {}
    for color in colors_raw:
        if 'first' in color and 'second' in color:
            color_name = color['first']['name']
            color_value = color['second']
        else:
            color_name = list(color.keys())[0]
            color_value = color[color_name]
        colors[color_name] = color_value

    tex_envs = {}
    for tex in tex_envs_raw:
        if 'first' in tex and 'second' in tex:
            tex_name = tex['first']['name']
            tex_value = tex['second']
        else:
            tex_name = list(tex.keys())[0]
            tex_value = tex[tex_name]
        tex_envs[tex_name] = tex_value

    floats = {}
    for float_prop in floats_raw:
        if 'first' in float_prop and 'second' in float_prop:
            float_name = float_prop['first']['name']
            float_value = float_prop['second']
        else:
            float_name = list(float_prop.keys())[0]
            float_value = float_prop[float_name]
        floats[float_name] = float_value

    return colors, tex_envs, floats

def copy_file_and_get_relative_path(source_file, destination_folder):
    """
    Copies a file to the destination folder and returns the relative path of the file in the destination folder.

    Parameters:
    - source_file: Absolute path of the source file
    - destination_folder: Path of the destination folder

    Returns:
    - relative_path: The relative path of the file in the destination folder
    """
    
    # Ensure the destination folder exists; if it does not, create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get the file name from the source file
    file_name = os.path.basename(source_file)

    # Construct the absolute path of the destination file
    destination_file = os.path.join(destination_folder, file_name)

    # Copy the file to the destination folder
    shutil.copy(source_file, destination_file)

    # Get the base name of the destination folder
    folder_basename = os.path.basename(destination_folder)

    # Construct the return path: 'folder_basename/relative_path'
    combined_path = os.path.join(folder_basename, file_name)

    return combined_path


# def add_texture(asset_id,prefab_info,root_path):
#     for sub_obj in prefab_info:
#         mesh_name = prefab_info[sub_obj]['MeshName']
#         mesh_path = prefab_info[sub_obj]['MeshPath']
#         mesh_file_extension = os.path.splitext(mesh_path)[1]
#         if mesh_file_extension == ".fbx":
#             obj_name = snake_to_camel(mesh_name)
#         elif mesh_file_extension == ".obj":
#             obj_name = snake_to_camel(asset_id)

#         source_dir = os.path.dirname(os.path.realpath(__file__))
#         texture_folder = os.path.join(source_dir, 'procthor_assets',f"{asset_id}", "textures")
#         if not os.path.exists(texture_folder):
#             os.makedirs(texture_folder)

#         material_slots = prefab_info[sub_obj]['Materials']
#         if len(material_slots) > 1:
#             # rewrite mtl file path
#             for i in range(len(material_slots)):
#                 mtl_filepath = os.path.join(source_dir, 'procthor_assets',f"{asset_id}", f"{obj_name}_GeomSubset_{i}.mtl")
#                 material_name = f"M_{obj_name}_GeomSubset_{i}"
#                 mat_filepath = os.path.join(root_path, prefab_info[sub_obj]['Materials'][i]['MaterialPath'])
            
#                 mat_file_extension = os.path.splitext(mat_filepath)[1]
 
#                 if mat_file_extension == ".mat":
#                     mat_data = load_mat_file(mat_filepath)
#                     single_mat = modify_mtl_file(material_name, mat_data, texture_folder, root_path)

#                 with open(mtl_filepath, 'w') as file:
#                     file.write(single_mat)

#         else:
#             mtl_filepath = os.path.join(source_dir, 'procthor_assets',f"{asset_id}", f"{obj_name}.mtl")

#             mat_list = []
#             material_name = f"M_{obj_name}"
#             mat_filepath = os.path.join(root_path, prefab_info[sub_obj]['Materials'][0]['MaterialPath'])

            
#             mat_file_extension = os.path.splitext(mat_filepath)[1]

#             if mat_file_extension == ".mat":
#                 mat_data = load_mat_file(mat_filepath)
#                 single_mat = modify_mtl_file(material_name, mat_data, texture_folder, root_path)
#                 # mat_list.append(single_mat)

#             # for i,mat_dict in enumerate(prefab_info[sub_obj]['Materials']):

#             #     material_name = f"M_{obj_name}_{i}"
#             #     mat_filepath = os.path.join(root_path, prefab_info[sub_obj]['Materials'][i]['MaterialPath'])
#             #     mat_file_extension = os.path.splitext(mat_filepath)[1]

#             #     if mat_file_extension == ".mat":
#             #         mat_data = load_mat_file(mat_filepath)
#             #         single_mat = modify_mtl_file(material_name, mat_data, texture_folder, root_path)
#             #         mat_list.append(single_mat)

#             # mat_content = "\n\n".join(mat_list)
#             with open(mtl_filepath, 'w') as file:
#                 file.write(single_mat)


def add_texture(asset_id,prefab_info,root_path):
    for sub_obj in prefab_info:
        mesh_name = prefab_info[sub_obj]['MeshName']
        mesh_path = prefab_info[sub_obj]['MeshPath']
        mesh_file_extension = os.path.splitext(mesh_path)[1]
        if mesh_file_extension == ".fbx":
            obj_name = snake_to_camel(mesh_name)
        elif mesh_file_extension == ".obj":
            obj_name = snake_to_camel(asset_id)

        source_dir = os.path.dirname(os.path.realpath(__file__))
        mtl_filepath = os.path.join(source_dir, 'procthor_assets',f"{asset_id}", f"{obj_name}.mtl")
        texture_folder = os.path.join(source_dir, 'procthor_assets',f"{asset_id}", "textures")

        if not os.path.exists(texture_folder):
            os.makedirs(texture_folder)
        
        mat_list = []

        for i,mat_dict in enumerate(prefab_info[sub_obj]['Materials']):

            material_name = f"M_{obj_name}_{i}"
            mat_filepath = os.path.join(root_path, prefab_info[sub_obj]['Materials'][i]['MaterialPath'])
            mat_file_extension = os.path.splitext(mat_filepath)[1]

            if mat_file_extension == ".mat":
                mat_data = load_mat_file(mat_filepath)
                single_mat = modify_mtl_file(material_name, mat_data, texture_folder, root_path)
                mat_list.append(single_mat)

        mat_content = "\n\n".join(mat_list)
        with open(mtl_filepath, 'w') as file:
            file.write(mat_content)

if __name__=="__main__":
    import json
    with open('/home/zgao/ai2thor/unity/Assets/AllPrefabDetails.json', 'r') as file:
        all_prefab_details = json.load(file)

    with open('/home/zgao/procthor/procthor/databases/asset-database.json', 'r') as file:
        procthor_database = json.load(file)

    root_path = '/home/zgao/ai2thor/unity'


    # # iterate from a specific key
    # start_key = 'Television'
    # keys = list(procthor_database.keys())
    # start_index = keys.index(start_key)
    # for asset_grp in keys[start_index:]:
    #     asset_list = procthor_database[asset_grp]
    #     for asset in asset_list:
    #         asset_id= asset['assetId']

    #         if asset_id in all_prefab_details:
    #             prefab_info = all_prefab_details[asset_id]
    #             convert_prefab(asset_id, prefab_info, root_path)
    #             add_texture(asset_id,prefab_info,root_path)

    #         else:
    #             print(f"Prefab {asset_id} not found in AllPrefabDetails.json")

    # for asset_grp in procthor_database:
    #     asset_list = procthor_database[asset_grp]
    #     for asset in asset_list:
    #         asset_id= asset['assetId']

    #         if asset_id in all_prefab_details:
    #             prefab_info = all_prefab_details[asset_id]
    #             convert_prefab(asset_id, prefab_info, root_path,shift_center=True)
    #             add_texture(asset_id,prefab_info,root_path)

    #         else:
    #             print(f"Prefab {asset_id} not found in AllPrefabDetails.json")

    with open('/home/zgao/ProcTHOR_Converter/house_7.json', 'r') as file:
        test_house = json.load(file)
    for obj in test_house['objects']:
        asset_id = obj['assetId']
        prefab_info = all_prefab_details[asset_id]
        convert_prefab(asset_id, prefab_info, root_path,shift_center=True)
        add_texture(asset_id,prefab_info,root_path)
        if "children" in obj:
            for child in obj["children"]:
                asset_id = child['assetId']
                prefab_info = all_prefab_details[asset_id]
                convert_prefab(asset_id, prefab_info, root_path,shift_center=True)
                add_texture(asset_id,prefab_info,root_path)


    # Test the conversion for a single prefab  
    # asset_id= 'Teddy_Bear_1'
    # asset_id = 'Vase_Tall_4'
    # asset_id = 'Toilet_1'
    # asset_id = 'Armchair_208_3'
    # Armchair_208_3
    # # asset_id = 'Plunger_3'
    # # asset_id = 'Toilet_Paper_Used_Up'
    # # asset_id = 'Toilet_Paper'
    # # asset_id = 'Bathroom_Faucet_27'
    # asset_id = 'Floor_Lamp_19'

    # prefab_info = all_prefab_details[asset_id]
    # convert_prefab(asset_id, prefab_info,root_path,shift_center=True)
    # process_pipeline(asset_id, prefab_info,root_path,shift_center=True)
    # add_texture(asset_id,prefab_info,root_path)
