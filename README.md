# Unity Prefab Converter

## Introduction
This project is to converte .prefab file to .obj/.stl files, and also converte procthor json files to usd.

prefab_to_folder.py is used to converte the prefab.

procthor_to_scene.py is used to converte json in procthor to usd. You need to modify the path of the converted raw assets in the import_asset_new() function.

AllPrefabDetails.json contains some important information of [ai2thor](https://github.com/allenai/ai2thor) assets in Unity, such as path of the material file, and it is generated by ParseAllPrefabs.cs.

asset-database.json is one configuration files of [Procthor](https://github.com/allenai/procthor/blob/main/procthor/databases/asset-database.json), contains the ai2thor resource file information used in procthor.

unity_mat_to_blender.py is only a test file. Some complex color effects can be implemented in blender and saved as .blende files, but unfortunately they cannot be saved to obj files.

In case I forget this

How to use ParseAllPrefabs.cs? 
Put it in ai2thor/unity/Assets/Editor folder, then we can find the Parse All Prefab in Assets option in the Tools drop-down menu at the top of the Unity interface.


## Update and TODOs
Some errors in the code:
The sub-object face indexes are in the wrong order.   Done

Some door meshes are overwritten due to duplicate names.  Done

Rewrite mesh names and files hierarchy?  Done

Fix the uvmap bugs. Done

Position of holes are not correct. Done

Paintings and windows need to be added.  windows done

Some objects' center and scale may be wrong

check the order of textures, might not match the order of sub-geomesh

Think about a mechanism, to avoid objects colliding with thick walls, e.g. paintings, TVs on the wall, bed close to the wall...

## Convert Pipelin
1. Prefab to  .obj file(s)
2. convert every .obj to .usda
3. split .usda according to texture
4. convert splited .usda to .obj


## NOTE: Doors and windows cannot share the same wall, per AI2-THOR limitation.
https://github.com/zhiyuan-gao/procthor/blob/4feb6a4f90afbc3d94448e8b3e4ae5727108d243/procthor/generation/wall_objects.py#L351C1-L351C83

