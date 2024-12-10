#!/usr/bin/env python3.11

import bpy
import os

def convert_usd_to_obj(usda_path, obj_path):

    bpy.ops.wm.read_factory_settings(use_empty=True)

    bpy.ops.wm.usd_import(filepath=usda_path)

    bpy.ops.wm.obj_export(filepath=obj_path, export_selected_objects=True, forward_axis='Y', up_axis='Z')

    print(f"Converted {usda_path} to {obj_path}")

# for i in range(0, 4):
#     usda_path = f"/home/zgao/unity_prefab_converter/procthor_assets/Floor_Lamp_19/M_FloorLamp_19_{i}.usda"  
#     obj_path = f"/home/zgao/unity_prefab_converter/procthor_assets/Floor_Lamp_19/M_FloorLamp_19_{i}.obj"       
#     convert_usd_to_obj(usda_path, obj_path)



if __name__=="__main__":
    import json

    with open('/home/zgao/procthor/procthor/databases/asset-database.json', 'r') as file:
        procthor_database = json.load(file)

    root_path = '/home/zgao/ai2thor/unity'
    source_dir = os.path.dirname(os.path.realpath(__file__))
    


    for asset_grp in procthor_database:
        asset_list = procthor_database[asset_grp]
        for asset in asset_list:
            asset_id= asset['assetId']
            assets_usd_dir = os.path.join(source_dir, 'procthor_assets', f'{asset_id}')
            usda_path = os.path.join(assets_usd_dir, f"{asset_id}.usda")
            
            
        