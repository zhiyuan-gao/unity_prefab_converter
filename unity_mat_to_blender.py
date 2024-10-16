import bpy
import yaml
import os

def find_asset_path_by_guid(root_path, guid):
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.meta'):
                meta_path = os.path.join(root, file)
                with open(meta_path, 'r') as meta_file:
                    if guid in meta_file.read():
                        return meta_path[:-5]  # remove ".meta"
    raise FileNotFoundError(f"No file with guid '{guid}' found in {root_path}")

def load_mat_file(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()
        content = [line for line in content if not line.startswith(('%YAML', '%TAG', '--- !u!'))]
        mat_content = yaml.safe_load("".join(content))
        mat_data = mat_content['Material']

    return mat_data


def get_or_create_node(nodes, node_type, node_name):
    node = nodes.get(node_name)
    if not node:
        node = nodes.new(type=node_type)
        node.name = node_name
    return node
# Unified function to apply any texture with scale and offset
def apply_texture_with_mapping(texture_path, scale, offset, nodes, links, bsdf=None, input_name=None):
    """
    General function to apply any texture with scale and offset.
    Optionally, connects the texture to a specified bsdf input if provided.
    """
    # Create Image Texture node
    tex_image = nodes.new(type='ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(texture_path)

    # Create Mapping and Texture Coordinate nodes
    mapping = get_or_create_node(nodes, 'ShaderNodeMapping', 'Mapping')
    mapping.inputs['Scale'].default_value = (scale['x'], scale['y'], 1)
    mapping.inputs['Location'].default_value = (offset['x'], offset['y'], 0)

    tex_coord = get_or_create_node(nodes, 'ShaderNodeTexCoord', 'Texture Coordinate')

    # Connect Mapping and Texture Coordinate nodes to Image Texture
    links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], tex_image.inputs['Vector'])

    # If bsdf and input_name are provided, connect the texture to the specified input
    if bsdf and input_name:
        links.new(tex_image.outputs['Color'], bsdf.inputs[input_name])
    
    return tex_image

# Function to apply Emission
def apply_emission(mat, texture_path=None, scale={'x': 1, 'y': 1}, offset={'x': 0, 'y': 0}, emission_color=(1.0, 1.0, 1.0, 1.0), strength=1.0):
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    emission_node = get_or_create_node(nodes, 'ShaderNodeEmission', 'Emission')
    emission_node.inputs['Color'].default_value = emission_color
    emission_node.inputs['Strength'].default_value = strength

    if texture_path:
        tex_image = apply_texture_with_mapping(texture_path, scale, offset, nodes, links)
        # Connect texture to Emission node
        links.new(tex_image.outputs['Color'], emission_node.inputs['Color'])

    mix_shader = get_or_create_node(nodes, 'ShaderNodeMixShader', 'Mix Shader')
    bsdf = nodes.get('Principled BSDF')
    material_output = nodes.get('Material Output')

    # Connect BSDF and Emission to Mix Shader
    links.new(bsdf.outputs['BSDF'], mix_shader.inputs[1])
    links.new(emission_node.outputs['Emission'], mix_shader.inputs[2])
    links.new(mix_shader.outputs['Shader'], material_output.inputs['Surface'])

# Function to apply Normal Map
def apply_normal_map(mat, texture_path, scale={'x': 1, 'y': 1}, offset={'x': 0, 'y': 0}):
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    tex_image = apply_texture_with_mapping(texture_path, scale, offset, nodes, links)
    tex_image.image.colorspace_settings.name = 'Non-Color'
    
    normal_map = get_or_create_node(nodes, 'ShaderNodeNormalMap', 'Normal Map')
    links.new(tex_image.outputs['Color'], normal_map.inputs['Color'])
    
    bsdf = nodes.get('Principled BSDF')
    links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])

# Function to apply Parallax Map (simulated using Bump Node)
def apply_parallax_map(mat, texture_path, scale={'x': 1, 'y': 1}, offset={'x': 0, 'y': 0}, strength=0.1):
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    tex_image = apply_texture_with_mapping(texture_path, scale, offset, nodes, links)
    tex_image.image.colorspace_settings.name = 'Non-Color'
    
    bump_node = get_or_create_node(nodes, 'ShaderNodeBump', 'Bump Map')
    bump_node.inputs['Strength'].default_value = strength
    links.new(tex_image.outputs['Color'], bump_node.inputs['Height'])
    
    bsdf = nodes.get('Principled BSDF')
    links.new(bump_node.outputs['Normal'], bsdf.inputs['Normal'])

# Function to apply Alpha Test (Alpha Clip)
def apply_alpha_test(mat, cutoff=0.5):
    mat.blend_method = 'CLIP'
    mat.shadow_method = 'CLIP'
    mat.alpha_threshold = cutoff

# Function to apply Alpha Blend
def apply_alpha_blend(mat):
    mat.blend_method = 'BLEND'
    mat.shadow_method = 'HASHED'

# Function to apply Alpha Premultiply
def apply_alpha_premultiply(mat):
    mat.blend_method = 'BLEND'
    mat.shadow_method = 'HASHED'

# Function to apply Detail Albedo Map
def apply_detail_albedo_map(mat, main_texture_path, detail_texture_path, scale={'x': 1, 'y': 1}, offset={'x': 0, 'y': 0}, blend_factor=0.5):
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    tex_image = apply_texture_with_mapping(main_texture_path, scale, offset, nodes, links)
    
    detail_tex_image = apply_texture_with_mapping(detail_texture_path, scale, offset, nodes, links)
    
    mix_rgb = get_or_create_node(nodes, 'ShaderNodeMixRGB', 'Mix RGB')
    mix_rgb.blend_type = 'MIX'
    mix_rgb.inputs['Fac'].default_value = blend_factor
    
    # Connect textures to Mix RGB node
    links.new(tex_image.outputs['Color'], mix_rgb.inputs['Color1'])
    links.new(detail_tex_image.outputs['Color'], mix_rgb.inputs['Color2'])
    
    bsdf = nodes.get('Principled BSDF')
    links.new(mix_rgb.outputs['Color'], bsdf.inputs['Base Color'])

# Function to apply Occlusion Map
def apply_occlusion_map(mat, texture_path, scale={'x': 1, 'y': 1}, offset={'x': 0, 'y': 0}):
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    tex_image = apply_texture_with_mapping(texture_path, scale, offset, nodes, links)
    tex_image.image.colorspace_settings.name = 'Non-Color'
    
    mix_rgb = get_or_create_node(nodes, 'ShaderNodeMixRGB', 'Mix RGB')
    mix_rgb.blend_type = 'MULTIPLY'
    
    bsdf = nodes.get('Principled BSDF')
    links.new(bsdf.inputs['Base Color'].default_value, mix_rgb.inputs['Color1'])
    links.new(tex_image.outputs['Color'], mix_rgb.inputs['Color2'])
    links.new(mix_rgb.outputs['Color'], bsdf.inputs['Base Color'])

# Function to apply Metallic Gloss Map
def apply_metallic_gloss_map(mat, texture_path, scale={'x': 1, 'y': 1}, offset={'x': 0, 'y': 0}):
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    tex_image = apply_texture_with_mapping(texture_path, scale, offset, nodes, links)
    
    separate_rgb = get_or_create_node(nodes, 'ShaderNodeSeparateRGB', 'Separate RGB')
    links.new(tex_image.outputs['Color'], separate_rgb.inputs['Image'])
    
    bsdf = nodes.get('Principled BSDF')
    links.new(separate_rgb.outputs['R'], bsdf.inputs['Metallic'])
    links.new(separate_rgb.outputs['G'], bsdf.inputs['Roughness'])

# Function to apply textures from m_TexEnvs
def apply_textures_from_texenvs(mat, tex_envs, root_path):
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get('Principled BSDF')
    
    for tex_env in tex_envs:
        for key, value in tex_env.items():
            texture_info = value.get('m_Texture', {})
            file_id = texture_info.get('fileID', 0)

            if file_id != 0:  # Ensure valid texture
                guid = texture_info.get('guid')
                texture_path = find_asset_path_by_guid(root_path, guid)
                scale = value.get('m_Scale', {'x': 1, 'y': 1})
                offset = value.get('m_Offset', {'x': 0, 'y': 0})

                if key == '_MainTex':
                    apply_texture_with_mapping(texture_path, scale, offset, nodes, links, bsdf, 'Base Color')
                elif key == '_BumpMap':
                    apply_normal_map(mat, texture_path, scale, offset)
                elif key == '_EmissionMap':
                    apply_emission(mat, texture_path, scale, offset)
                elif key == '_MetallicGlossMap':
                    apply_metallic_gloss_map(mat, texture_path, scale, offset)
                elif key == '_OcclusionMap':
                    apply_occlusion_map(mat, texture_path, scale, offset)
                elif key == '_DetailAlbedoMap':
                    detail_texture_path = find_asset_path_by_guid(root_path, value.get('guid', ''))
                    apply_detail_albedo_map(mat, texture_path, detail_texture_path, scale, offset)
                elif key == '_ParallaxMap':
                    apply_parallax_map(mat, texture_path, scale, offset)

# Main function to process m_ShaderKeywords and m_TexEnvs
def setup_material(mat, mat_data, root_path):
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Get or create Principled BSDF node
    bsdf = get_or_create_node(nodes, 'ShaderNodeBsdfPrincipled', 'Principled BSDF')

    # Get or create Material Output node
    material_output = get_or_create_node(nodes, 'ShaderNodeOutputMaterial', 'Material Output')

    # Apply Shader Keywords
    shader_keywords = mat_data.get('m_ShaderKeywords', '')

    if '_EMISSION' in shader_keywords:
        emission_color = mat_data.get('_EmissionColor', (1.0, 1.0, 1.0, 1.0))
        apply_emission(mat, emission_color)

    if '_NORMALMAP' in shader_keywords:
        normal_map_path = find_asset_path_by_guid(root_path, mat_data.get('NormalMapGuid', ''))
        apply_normal_map(mat, normal_map_path)

    if '_ALPHATEST_ON' in shader_keywords:
        apply_alpha_test(mat)

    if '_ALPHABLEND_ON' in shader_keywords:
        apply_alpha_blend(mat)

    if '_ALPHAPREMULTIPLY_ON' in shader_keywords:
        apply_alpha_premultiply(mat)

    if '_PARALLAXMAP' in shader_keywords:
        parallax_map_path = find_asset_path_by_guid(root_path, mat_data.get('ParallaxMapGuid', ''))
        apply_parallax_map(mat, parallax_map_path)

    if '_DETAIL_MULX2' in shader_keywords:
        main_texture_path = find_asset_path_by_guid(root_path, mat_data.get('MainTexGuid', ''))
        detail_texture_path = find_asset_path_by_guid(root_path, mat_data.get('DetailAlbedoMapGuid', ''))
        apply_detail_albedo_map(mat, main_texture_path, detail_texture_path)

    if '_OCCLUSIONMAP' in shader_keywords:
        occlusion_map_path = find_asset_path_by_guid(root_path, mat_data.get('OcclusionMapGuid', ''))
        apply_occlusion_map(mat, occlusion_map_path)

    if '_METALLICGLOSSMAP' in shader_keywords:
        metallic_gloss_map_path = find_asset_path_by_guid(root_path, mat_data.get('MetallicGlossMapGuid', ''))
        apply_metallic_gloss_map(mat, metallic_gloss_map_path)

    # Apply Textures from m_TexEnvs
    tex_envs = mat_data.get('m_SavedProperties', {}).get('m_TexEnvs', [])
    apply_textures_from_texenvs(mat, tex_envs, root_path)





if __name__=="__main__":
        
    import json

    def import_obj(obj_file_path):
        # bpy.ops.import_scene.obj(filepath=file_path)
        bpy.ops.wm.obj_import(filepath=obj_file_path)
        
        imported_object = bpy.context.selected_objects[0]
        return imported_object

    def apply_material_to_object(obj, mat):
        if obj.data.materials:
            
            obj.data.materials[0] = mat
        else:
            
            obj.data.materials.append(mat)







    with open('/home/zgao/ProcTHOR_Converter/AllPrefabDetails.json', 'r') as file:
        all_prefab_details = json.load(file)

    with open('/home/zgao/procthor/procthor/databases/asset-database.json', 'r') as file:
        procthor_database = json.load(file)
    # for asset_grp in procthor_database:
    #     asset_list = procthor_database[asset_grp]
    #     for asset in asset_list:
    #         asset_id= asset['assetId']
    #         convert_prefab(asset_id, all_prefab_details)

    for asset_grp in procthor_database:
        asset_list = procthor_database[asset_grp]
        for asset in asset_list:
            asset_id= asset['assetId']
            obj_folder = f"/home/zgao/ProcTHOR_Converter/procthor_assets/{asset_id}"
            # for file in os.listdir(obj_folder):
            #     if file.endswith(".obj"):



    root_path='/home/zgao/ai2thor/unity/Assets'

    # mat_file_path = "/home/zgao/ai2thor/unity/Assets/Resources/QuickMaterials/Fabrics/Carpet7.mat"
    mat_file_path = '/home/zgao/ai2thor/unity/Assets/Resources/Living Room Objects/Television/Materials/Television_Primary1_Mat.mat'

    obj_folder = "/home/zgao/ProcTHOR_Converter/procthor_assets"



    # obj_file_path = "/home/zgao/ProcTHOR_Converter/procthor_assets/Bed_28/Bed_28.obj"
    obj_file_path = "/home/zgao/ProcTHOR_Converter/procthor_assets/Television_1/Television_1.obj"
    

    mat_data = load_mat_file(mat_file_path)

    # Create a new material and apply the settings
    mat = bpy.data.materials.new(name="UnityMaterial")
    setup_material(mat, mat_data, root_path)
    imported_object = import_obj(obj_file_path)

    apply_material_to_object(imported_object, mat)



    save_file_path = "/home/zgao/ProcTHOR_Converter/procthor_assets/Television_1/Television_1.blend"
    bpy.ops.wm.save_as_mainfile(filepath=save_file_path)