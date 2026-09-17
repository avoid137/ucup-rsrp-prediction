r"""
generate_simulation_data.py
===========================
步骤 2/4：用物理公式给每个基站"造"一批带标签的训练样本。

动机：真实实测样本虽然有百万级，但只覆盖已建站的实际场景，分布受限；
模型很难仅靠它学会「距离—损耗」「遮挡—衰减」这类物理规律。所以先用
自由空间路径损耗 + 建筑遮挡 + 天线增益 + 多径的解析公式合成一批数据做预训练，
再用更小的学习率在真实数据上微调（见 two_stage_training.py）。

每 (基站, 采样点) 生成一条样本，标签是公式算出来的 RSRP：
    RSRP = 发射功率 - 总路径损耗
    总路径损耗 = 自由空间损耗 + 遮挡损耗 - 天线增益 - 多径增益 + 噪声

遮挡判定用点云网格顶点的 KDTree：沿收发连线采样 30 个位置，任一位置距
最近墙体顶点小于 3 m 即判为遮挡。单次查询是 O(log N) 而不是 O(N)，
这是整条流水线能跑完的关键。

注意：这里算出的 is_blocked 等中间量只是**标签的中间结果**，不进特征表 ——
实测数据拿不到遮挡真值，特征里两阶段统一用几何代理（见 features.py）。
遮挡信息通过标签间接进入模型。

依赖：open3d, numpy, pandas, scipy
用法：
    set UCUP_DATA_ROOT=D:\path\to\TrainingData.26UCupSummer
    set UCUP_MESH_DIR=D:\path\to\TrainingData_Meshes_v2
    python generate_simulation_data.py
"""

import json
import os
import warnings
import zlib

import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import KDTree

from features import RECEIVER_HEIGHT

warnings.filterwarnings("ignore")

# ==================== 配置 ====================
DATA_ROOT = os.environ.get("UCUP_DATA_ROOT", "data/TrainingData.26UCupSummer")
MESH_DIR = os.environ.get("UCUP_MESH_DIR", "data/TrainingData_Meshes_v2")
OUTPUT_DIR = os.environ.get("UCUP_OUTPUT_DIR", "outputs/simulation_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FREQUENCY = 3.5e9        # 载波频率 3.5 GHz
TX_POWER_DBM = 46.0      # 发射功率
SAMPLES_PER_CELL = 2000  # 每个基站生成样本数
MAX_DISTANCE = 500       # 最大采样距离（米）
RAY_SAMPLES = 30         # 遮挡判定沿连线的采样点数
WALL_DISTANCE = 3.0      # 判定为遮挡的墙体距离阈值（米）


def cell_seed(cell_name):
    """与运行环境无关的确定性随机种子。

    注意别用 Python 内置 hash()：字符串 hash 受 PYTHONHASHSEED 影响，
    每个进程都不一样，同一份代码两次运行会生成不同的"仿真数据"，
    结果就没法复现了。crc32 是确定的。
    """
    return zlib.crc32(cell_name.encode("utf-8")) % (2 ** 32)


# ==================== 核心模拟函数 ====================
def simulate_cell(cell_name, mesh_path, ep_path, num_samples=SAMPLES_PER_CELL):
    """对单个基站做物理仿真，生成带标签的样本。"""
    print(f"   模拟: {cell_name}")

    try:
        with open(ep_path, "r") as f:
            ep = json.load(f)
    except Exception:
        print("      无法读取 ep.json")
        return None

    antenna_height = ep.get("height", 30.0)
    azimuth = ep.get("azimuth", 0.0)
    downtilt = ep.get("downtilt", 5.0)

    # 读取网格并建 KDTree（用于遮挡检测）
    try:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if mesh is None or len(mesh.vertices) == 0:
            print("      网格为空，跳过")
            return None
        vertices = np.asarray(mesh.vertices)
        tree = KDTree(vertices)
    except Exception as e:
        print(f"      网格读取失败: {e}")
        return None

    np.random.seed(cell_seed(cell_name))
    tx_pos = np.array([0.0, 0.0, antenna_height])
    samples = []
    attempts = 0
    max_attempts = num_samples * 4

    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1

        # 距离采样：指数分布，让样本偏向近处（近处信号强、更值得学）
        r = min(np.random.exponential(scale=120), MAX_DISTANCE)
        if r < 10:
            continue

        # 角度采样：60% 落在主瓣附近（正态），40% 全向均匀
        if np.random.random() < 0.6:
            theta = np.random.normal(azimuth, 30)
        else:
            theta = np.random.uniform(0, 360)
        theta = theta % 360

        x = r * np.cos(np.radians(theta))
        y = r * np.sin(np.radians(theta))
        rx_pos = np.array([x, y, RECEIVER_HEIGHT])

        dist = float(np.linalg.norm(rx_pos - tx_pos))
        if dist < 10:
            continue

        # ---------- 物理计算 ----------
        # 1. 自由空间路径损耗（Friis 公式的 dB 形式）
        pl_free = 20 * np.log10(dist) + 20 * np.log10(FREQUENCY) - 147.55

        # 2. 建筑遮挡：沿收发连线采样，任一采样点距墙体过近即判为遮挡
        is_blocked = False
        blockage_loss = 0.0
        for t in np.linspace(0.05, 0.95, RAY_SAMPLES):
            d_wall, _ = tree.query(tx_pos + t * (rx_pos - tx_pos), k=1)
            if d_wall < WALL_DISTANCE:
                is_blocked = True
                blockage_loss = min(30.0, 10.0 + 20.0 * (1 - d_wall / WALL_DISTANCE))
                break

        # 3. 天线增益：主瓣内二次衰减，主瓣外线性衰减
        angle_to_rx = np.arctan2(y, x) * 180 / np.pi
        azimuth_diff = abs(angle_to_rx - azimuth)
        azimuth_diff = min(azimuth_diff, 360 - azimuth_diff)
        if azimuth_diff < 30:
            antenna_gain = -3 * (azimuth_diff / 30) ** 2
        else:
            antenna_gain = -15 - (azimuth_diff - 30) / 5

        # 4. 多径增益：视距与非视距分布不同
        if not is_blocked:
            multipath_gain = np.random.uniform(-3, 3)
            if dist < 50:
                multipath_gain += np.random.uniform(1, 3)
        elif np.random.random() < 0.25:
            multipath_gain = np.random.uniform(-8, -3)
        else:
            multipath_gain = np.random.uniform(-20, -8)

        # 5. 阴影衰落噪声（简化：高斯；真实城市宏小区是对数正态 σ≈4–12 dB）
        noise = np.random.normal(0, 1.5)

        # 6. 总路径损耗 -> RSRP
        pl_total = pl_free + blockage_loss - antenna_gain - multipath_gain + noise
        rsrp = float(np.clip(TX_POWER_DBM - pl_total, -130, -40))

        samples.append({
            "cell_id": cell_name,
            "x": float(x),
            "y": float(y),
            "z": RECEIVER_HEIGHT,
            "is_blocked": int(is_blocked),
            "blockage_loss": float(blockage_loss),
            "pl_free": float(pl_free),
            "pl_total": float(pl_total),
            "rsrp": rsrp,
            "antenna_height": antenna_height,
            "azimuth": azimuth,
            "downtilt": downtilt,
        })

    if len(samples) < num_samples * 0.5:
        print(f"      只生成了 {len(samples)} 个样本 (目标 {num_samples})")

    return samples


def generate_all_simulation_data():
    print("=" * 60)
    print("  步骤 2/4: 生成仿真数据")
    print("=" * 60)

    if not os.path.isdir(DATA_ROOT):
        raise SystemExit(f"数据目录不存在: {DATA_ROOT}\n请设置环境变量 UCUP_DATA_ROOT")

    # 判定标准是"有 ep.json"（配合网格是否就绪），不靠目录名前缀
    cell_dirs = []
    for name in sorted(os.listdir(DATA_ROOT)):
        path = os.path.join(DATA_ROOT, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "ep.json")):
            cell_dirs.append(path)
    print(f"找到 {len(cell_dirs)} 个基站")

    all_samples = []
    for i, cell_dir in enumerate(cell_dirs, 1):
        cell_name = os.path.basename(cell_dir)
        mesh_path = os.path.join(MESH_DIR, f"{cell_name}_mesh.obj")
        if not os.path.exists(mesh_path):
            print(f"\n[{i}/{len(cell_dirs)}] {cell_name}: 网格不存在，跳过")
            continue

        print(f"\n[{i}/{len(cell_dirs)}] 处理: {cell_name}")
        samples = simulate_cell(
            cell_name, mesh_path, os.path.join(cell_dir, "ep.json"), SAMPLES_PER_CELL
        )
        if samples:
            all_samples.extend(samples)
            print(f"      生成 {len(samples)} 个样本")
        else:
            print("      模拟失败")

    if not all_samples:
        print("\n没有生成任何样本")
        return None

    df = pd.DataFrame(all_samples)
    output_path = os.path.join(OUTPUT_DIR, "simulation_data.csv")
    df.to_csv(output_path, index=False)
    print(f"\n仿真数据已保存: {output_path}")
    print(f"   总样本数: {len(df):,}")
    print(f"   基站数:   {df['cell_id'].nunique()}")
    print(f"   RSRP 范围: {df['rsrp'].min():.1f} ~ {df['rsrp'].max():.1f} dBm")
    return df


if __name__ == "__main__":
    generate_all_simulation_data()
