#!/usr/bin/env python
from omni.isaac.kit import SimulationApp

# Headless 模式启动 Isaac Sim
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics
import os
import re

def is_static_object(name: str) -> bool:
    """
    根据物体名称判断是否是静态物体，不需要添加物理属性
    """
    return bool(re.search(r"(wall|painting|window|ground|television|door|fridge|DiningTable)", name.lower()))

def get_xform_prims_under(stage, root_path: str):
    """
    遍历 stage 中 root_path 下的一级 Xform 节点
    """
    xform_prims = []
    root = stage.GetPrimAtPath(root_path)
    if not root:
        print(f"未找到根节点: {root_path}")
        return xform_prims
    for child in root.GetChildren():
        if child.GetTypeName() == "Xform":
            xform_prims.append(child)
    return xform_prims

def apply_physics_to_xforms(source_usd_path: str, root_prim="/house_9", default_mass=10.0):
    """
    为 USD 文件中 root_prim 下的所有物体（Xform 节点）添加物理属性，
    并另存为一个新文件，不覆盖原始文件
    """
    # 打开 USD 文件
    stage = Usd.Stage.Open(source_usd_path)
    if not stage:
        print(f"无法打开: {source_usd_path}")
        return

    print(f"📂 已打开 USD 文件: {source_usd_path}")
    

    # 在根节点下创建一个新的 Xform 作为地面碰撞体
    floor_path = "/house_2/GroundCollision"
    floor_xform = UsdGeom.Xform.Define(stage, floor_path)

    # 定义一个 Cube 作为地面碰撞体
    floor_mesh = UsdGeom.Cube.Define(stage, floor_path + "/CollisionGeo")

    # 使用 USD API 添加 Translate 和 Scale 操作
    # 对于上轴为 Z，Cube 默认中心在 (0,0,0) 且尺寸为 1 单位
    # 我们希望立方体高度为 0.1，且上表面位于 Z=0，所以需要将立方体沿 Z 轴平移 -0.05
    translate_op = floor_mesh.AddTranslateOp()
    translate_op.Set((0, 0, -0.15))

    # 设置缩放：X和Y方向延展到20单位，高度缩放到 0.1 单位
    scale_op = floor_mesh.AddScaleOp()
    scale_op.Set((10, 10, 0.1))

    # 为地面 Xform 添加碰撞属性（物理仿真用）
    prim = floor_xform.GetPrim()
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)





    # 获取所有一级 Xform 节点（代表物体）
    xform_prims = get_xform_prims_under(stage, root_prim)
    print(f"🔍 在 {root_prim} 下找到 {len(xform_prims)} 个物体 (Xform)")

    for prim in xform_prims:
        name = prim.GetName()
        path = prim.GetPath().pathString
        if is_static_object(name):
            print(f"⏭️ 跳过静态物体: {path}")
            continue

        print(f"⚙️ 处理物体: {path}")

        # 添加刚体（RigidBody）— 使用 prim.HasAPI() 检查是否已添加
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(prim)
        # 添加碰撞体（Collision）
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
        # 添加质量（Mass），默认值为 default_mass
        if not prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(default_mass)

    # 构造新文件路径，不覆盖原始文件
    folder, filename = os.path.split(source_usd_path)
    name_no_ext, ext = os.path.splitext(filename)
    new_filename = f"{name_no_ext}_with_physics{ext}"
    new_usd_path = os.path.join(folder, new_filename)

    # 将修改后的 stage 导出为新文件
    stage.GetRootLayer().Export(new_usd_path)
    print(f"✅ 场景已处理完毕，另存为: {new_usd_path}")

# ======= 主程序入口 =======
if __name__ == "__main__":
    # 替换为你的原始 USD 文件的绝对路径
    source_path = "/home/zgao/unity_prefab_converter/house_2/house_2.usda"
    apply_physics_to_xforms(source_path, root_prim="/house_2", default_mass=10.0)
    
    # 关闭 Isaac Sim 模拟环境
    simulation_app.close()