r"""
pointcloud_to_mesh.py
=====================
步骤 1/4：把每个基站的原始点云（.ply）重建成 3D 网格（.obj），供后续遮挡判定使用。

做法：裁剪地面以下点 -> 体素降采样 -> 估计法线 -> 泊松重建 -> 按密度分位剪枝。
泊松重建把散乱点云变成封闭曲面，之后才能用 KDTree 做"射线是否穿墙"的近邻查询。

依赖：open3d, numpy
用法：
    set UCUP_DATA_ROOT=D:\path\to\TrainingData.26UCupSummer
    set UCUP_MESH_DIR=D:\path\to\TrainingData_Meshes_v2
    python pointcloud_to_mesh.py
"""

import glob
import json
import os

import numpy as np
import open3d as o3d

# ==================== 配置 ====================
# 路径都可以用环境变量覆盖，默认按仓库内的相对目录组织。
DATA_ROOT = os.environ.get("UCUP_DATA_ROOT", "data/TrainingData.26UCupSummer")
OUTPUT_DIR = os.environ.get("UCUP_MESH_DIR", "data/TrainingData_Meshes_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 重建参数
VOXEL_SIZE = 0.5       # 体素边长（米）
DEPTH = 8              # 泊松重建八叉树深度，越大越细也越吃内存
POINT_LIMIT = 300000   # 降采样目标点数上限


def estimate_ground_z(points, antenna_height):
    """估算地面高度。

    点云不以绝对海拔表达，拿不到真实地面高程，所以用两个启发式量折中：
      a) 天线挂高的 0.7 倍（假设天线大致在建筑高度附近、z 轴以天线为参考）
      b) 点云 z 的 5% 分位（建筑立面之外的散乱低点）
    两者相差过大（> 20 m）说明启发式 a 明显不成立，就退回用数据自己的分位数。
    """
    z_vals = points[:, 2]
    ground_from_data = float(np.percentile(z_vals, 5))
    ground_from_height = -antenna_height * 0.7
    if abs(ground_from_height - ground_from_data) > 20:
        return ground_from_data
    return (ground_from_height + ground_from_data) / 2


def process_cell(cell_dir, output_dir):
    """处理单个基站：点云 -> 网格。"""
    cell_name = os.path.basename(cell_dir)
    print(f"\n[处理] {cell_name}")

    # 读取 ep.json 拿天线高度，用来估计地面高度
    ep_path = os.path.join(cell_dir, "ep.json")
    if not os.path.exists(ep_path):
        print("   [跳过] 找不到 ep.json")
        return False

    with open(ep_path, "r") as f:
        ep = json.load(f)
    antenna_height = ep.get("height", 30)

    # 查找 PLY 文件（排序后取第一个，保证同一份数据多次运行选择一致）
    ply_files = sorted(glob.glob(os.path.join(cell_dir, "*.ply")))
    if not ply_files:
        print("   [跳过] 找不到 PLY 文件")
        return False
    if len(ply_files) > 1:
        print(f"   [提示] 发现 {len(ply_files)} 个 PLY，使用 {os.path.basename(ply_files[0])}")

    try:
        pcd = o3d.io.read_point_cloud(ply_files[0])
    except Exception as e:
        print(f"   [跳过] 读取失败: {e}")
        return False

    points = np.asarray(pcd.points)
    if len(points) == 0:
        print("   [跳过] 点云为空")
        return False
    print(f"   原始点数: {len(points):,}, 天线高度: {antenna_height} m")

    # 去掉地面点，只留建筑立面
    ground_z = estimate_ground_z(points, antenna_height)
    filtered_points = points[points[:, 2] > ground_z + 1.0]
    print(f"   过滤后: {len(filtered_points):,} 点（地面估计 z = {ground_z:.1f}）")
    if len(filtered_points) == 0:
        print("   [跳过] 过滤后无剩余点")
        return False

    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)

    # 降采样：按点数超限比例反推体素尺寸，保持几何尺度一致
    if len(pcd_filtered.points) > POINT_LIMIT:
        voxel = VOXEL_SIZE * (len(pcd_filtered.points) / POINT_LIMIT) ** (1 / 3)
        pcd_filtered = pcd_filtered.voxel_down_sample(voxel_size=voxel)
        print(f"   体素降采样 {voxel:.2f} m -> {len(pcd_filtered.points):,} 点")

    # 法线估计（泊松重建的必需输入）
    pcd_filtered.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=VOXEL_SIZE * 5, max_nn=30
        )
    )

    # 泊松重建
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_filtered, depth=DEPTH, linear_fit=True
    )

    # 按密度分位剪掉低置信度的"外扩"面片
    densities = np.asarray(densities)
    if len(densities) > 0:
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.05))

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    output_path = os.path.join(output_dir, f"{cell_name}_mesh.obj")
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"   [完成] {output_path}  顶点 {len(mesh.vertices):,} / 面 {len(mesh.triangles):,}")
    return True


def main():
    print("=" * 60)
    print("  步骤 1/4: 点云 -> 3D 网格")
    print("=" * 60)

    if not os.path.isdir(DATA_ROOT):
        raise SystemExit(f"数据目录不存在: {DATA_ROOT}\n请设置环境变量 UCUP_DATA_ROOT")

    cell_dirs = []
    for name in sorted(os.listdir(DATA_ROOT)):
        path = os.path.join(DATA_ROOT, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "ep.json")):
            cell_dirs.append(path)
    print(f"找到 {len(cell_dirs)} 个基站")

    success = 0
    for i, cell_dir in enumerate(cell_dirs, 1):
        print(f"\n[{i}/{len(cell_dirs)}]")
        if process_cell(cell_dir, OUTPUT_DIR):
            success += 1

    print(f"\n完成: {success}/{len(cell_dirs)}")


if __name__ == "__main__":
    main()
