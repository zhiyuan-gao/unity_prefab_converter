#!/usr/bin/env python
from omni.isaac.kit import SimulationApp

# Start Isaac Sim in headless mode
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics
import os
import re

def is_static_object(name: str) -> bool:
    """
    Determine whether an object is static based on its name,
    meaning it does not need dynamic physics properties.
    """
    return bool(re.search(r"(wall|painting|window|floor|television|door|fridge|diningtable)", name.lower()))

def get_xform_prims_under(stage, root_path: str):
    """
    Traverse the first-level Xform nodes under root_path in the stage.
    """
    xform_prims = []
    root = stage.GetPrimAtPath(root_path)
    if not root:
        print(f"Root node not found: {root_path}")
        return xform_prims
    for child in root.GetChildren():
        if child.GetTypeName() == "Xform":
            xform_prims.append(child)
    return xform_prims

def apply_physics_to_xforms(source_usd_path: str, root_prim="/house_9", default_mass=10.0):
    """
    Add physics properties to all objects (Xform nodes) under root_prim in the USD file,
    and save as a new file without overwriting the original.
    For static objects (determined by is_static_object), only collision properties are added.
    For dynamic objects, RigidBody, Collision, and Mass properties are added.
    """
    # Open the USD file
    stage = Usd.Stage.Open(source_usd_path)
    if not stage:
        print(f"Unable to open: {source_usd_path}")
        return

    print(f"📂 USD file opened: {source_usd_path}")
    
    # Get all first-level Xform nodes (representing objects)
    xform_prims = get_xform_prims_under(stage, root_prim)
    print(f"🔍 Found {len(xform_prims)} objects (Xform) under {root_prim}")

    for prim in xform_prims:
        name = prim.GetName()
        path = prim.GetPath().pathString
        if is_static_object(name):
            print(f"⏭ Static object detected: {path}")
            # Only add collision properties for static objects.
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            continue

        print(f"Processing dynamic object: {path}")

        # Add a rigid body (RigidBody) for dynamic objects.
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(prim)
        # Add collision properties (Collision).
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
        # Add mass (Mass) with default value default_mass.
        if not prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(default_mass)

    # Construct new file path without overwriting the original file.
    folder, filename = os.path.split(source_usd_path)
    name_no_ext, ext = os.path.splitext(filename)
    new_filename = f"{name_no_ext}_with_physics{ext}"
    new_usd_path = os.path.join(folder, new_filename)

    # Export the modified stage as a new file.
    stage.GetRootLayer().Export(new_usd_path)
    print(f"✅ Scene processed and saved as: {new_usd_path}")

# ======= Main Program Entry =======
if __name__ == "__main__":
    # Replace with the absolute path of your original USD file.
    source_path = "/home/zgao/unity_prefab_converter/house_2/house_2.usda"
    apply_physics_to_xforms(source_path, root_prim="/house_2", default_mass=10.0)
    
    # Close the Isaac Sim simulation environment.
    simulation_app.close()
