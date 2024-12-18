import numpy as np
from math import atan2, asin, degrees

def transformation_matrix(position, rotation_deg, scale):
    """ 生成 4x4 变换矩阵 """
    rx, ry, rz = np.radians(rotation_deg)  # 转换为弧度
    sx, sy, sz = scale
    px, py, pz = position

    # 旋转矩阵（绕 X, Y, Z 轴）
    Rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rx), -np.sin(rx), 0],
                   [0, np.sin(rx), np.cos(rx), 0],
                   [0, 0, 0, 1]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry), 0],
                   [0, 1, 0, 0],
                   [-np.sin(ry), 0, np.cos(ry), 0],
                   [0, 0, 0, 1]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0, 0],
                   [np.sin(rz), np.cos(rz), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # 缩放矩阵
    S = np.array([[sx, 0, 0, 0],
                  [0, sy, 0, 0],
                  [0, 0, sz, 0],
                  [0, 0, 0, 1]])

    # 平移矩阵
    T = np.array([[1, 0, 0, px],
                  [0, 1, 0, py],
                  [0, 0, 1, pz],
                  [0, 0, 0, 1]])

    # 综合变换矩阵: T * Rz * Ry * Rx * S
    return T @ Rz @ Ry @ Rx @ S

# 第一个变换矩阵
position1 = (-3, 0, -4)
rotation1 = (0, 0, 0)
scale1 = (1, 1, 1)

# 第二个变换矩阵
position2 = (-0.00194823, 0.2734792, -0.1429131)
rotation2 = (159.54, 0, 0)
scale2 = (1, 1, 1)

# 计算两个变换矩阵
M1 = transformation_matrix(position1, rotation1, scale1)
M2 = transformation_matrix(position2, rotation2, scale2)

# 矩阵相乘: M_result = M1 * M2
# M_result = M1 @ M2

M_result = M2 @ M1


def extract_translation_and_rotation(matrix):
    """
    从 4x4 变换矩阵中提取位移向量和 XYZ 欧拉角
    """
    # 提取位移向量
    translation = matrix[:3, 3]
    
    # 提取旋转分量（前 3x3 矩阵）
    R = matrix[:3, :3]

    # 从旋转矩阵计算 XYZ 欧拉角
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = atan2(R[2, 1], R[2, 2])  # 绕 X 轴旋转
        y = atan2(-R[2, 0], sy)      # 绕 Y 轴旋转
        z = atan2(R[1, 0], R[0, 0])  # 绕 Z 轴旋转
    else:
        x = atan2(-R[1, 2], R[1, 1])
        y = atan2(-R[2, 0], sy)
        z = 0

    # 将弧度转换为角度
    euler_angles = [degrees(x), degrees(y), degrees(z)]

    return translation, euler_angles

translation, euler_angles = extract_translation_and_rotation(M_result)

print("位移向量 (Translation):", translation)
print("XYZ 欧拉角 (Euler Angles in degrees):", euler_angles)



# 位移向量 (Translation): [-3.00194823  0.2734792  -4.1429131 ]
# XYZ 欧拉角 (Euler Angles in degrees): [159.54, -0.0, 0.0]




# 位移向量 (Translation): [-3.00194823  1.6716927   3.60475271]
# XYZ 欧拉角 (Euler Angles in degrees): [159.54, -0.0, 0.0]