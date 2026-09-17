"""
features.py
===========
特征工程的**唯一实现**，被训练脚本（two_stage_training.py）、推理脚本
（generate_submission.py）与仿真器（generate_simulation_data.py）共同引用。

为什么要单独抽出来：最初版本把同一段特征构造在两个脚本里各抄了一份，
训练用的特征和推理用的特征一旦不一致（哪怕只是某一列的定义变了），
预测结果会断崖式下跌。抽成公共模块后，这类漂移在结构上不可能发生。

17 维特征（顺序即 FEATURE_COLS，训练与推理必须完全一致）：

    x, y                     接收点水平坐标（基站为原点）
    distance, distance_sq,   水平距离及其平方 / log / 倒数
    distance_log, distance_inv
    angle, sin_angle,        方位角（统一 [0,360)）及其三角函数
    cos_angle
    antenna_height,          基站工程参数：天线挂高 / 方位角 / 下倾角
    azimuth, downtilt
    height_dist_ratio,       天线与距离的交互项
    downtilt_dist
    azimuth_diff,            接收点与主瓣的方位角差、天线增益代理
    antenna_gain
    blocked_feature          遮挡代理特征（几何可得，两阶段同源）
"""

import numpy as np

# 接收点高度（米）。仿真器的收发几何与这里的距离定义共用同一个常量。
RECEIVER_HEIGHT = 1.5

FEATURE_COLS = [
    "x", "y",
    "distance", "distance_sq", "distance_log", "distance_inv",
    "angle", "sin_angle", "cos_angle",
    "antenna_height", "azimuth", "downtilt",
    "height_dist_ratio", "downtilt_dist",
    "azimuth_diff", "antenna_gain",
    "blocked_feature",
]


def get_feature_cols():
    """返回特征列名（副本，避免调用方误改全局）。"""
    return list(FEATURE_COLS)


def engineer_features(df, eps=1.0):
    """从原始列 x、y、antenna_height、azimuth、downtilt 构造 17 维特征。

    df 需至少包含 x、y、antenna_height、azimuth、downtilt 五列。

    与最初版本的三处差异（都是为了两个阶段真正同源）：

    1. **distance / angle 一律现算**，不再"列已存在就沿用"。
       仿真表里的 distance 是含天线高度差的 3D 距离，实测表里是水平距离，
       同名不同义会让阶段 1 学到的分裂点（threshold）在阶段 2 失去物理含义。
       这里统一取水平距离；高度差带来的偏差量级可忽略
       （500 m、30 m 挂高时 3D 距离仅相差 0.16%，折算到路损约 0.014 dB）。

    2. **angle 统一到 [0, 360)**。arctan2 直接输出 (-180, 180]，而仿真器采样
       的角度是 [0, 360)，同一方向会取到两个相差 360 的值，树模型按阈值分裂时
       会错位。sin/cos 两列本来就不受这个影响，但 angle 本身是入模特征。

    3. **blocked_feature 两个阶段都用几何代理**。实测数据没有遮挡真值，
       若仿真用 is_blocked 真值、实测用距离衰减代理，同一列在两张表里就不是
       同一个量，阶段之间的迁移也就无从谈起。仿真的遮挡信息仍然通过标签
       （rsrp 由 KDTree 遮挡判定参与计算）间接进入模型。
    """
    df = df.copy()

    # ---- 几何量：统一从 x、y 现算 ----
    df["distance"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
    df["angle"] = (np.arctan2(df["y"], df["x"]) * 180.0 / np.pi) % 360.0

    # 距离特征：路径损耗在 dB 域是 log 关系，所以三种变换都给模型
    df["distance_sq"] = df["distance"] ** 2
    df["distance_log"] = np.log(df["distance"] + eps)
    df["distance_inv"] = 1.0 / (df["distance"] + eps)

    # 角度特征：三角函数化，避免 0/360 处不连续
    df["angle_rad"] = np.radians(df["angle"])
    df["sin_angle"] = np.sin(df["angle_rad"])
    df["cos_angle"] = np.cos(df["angle_rad"])

    # 天线与距离的交互项
    df["height_dist_ratio"] = df["antenna_height"] / (df["distance"] + eps)
    df["downtilt_dist"] = df["downtilt"] * df["distance"] / 100.0

    # 方位角差与天线增益代理（主瓣 ±30° 的高斯型）
    df["azimuth_diff"] = np.abs(df["angle"] - df["azimuth"]) % 360.0
    df["azimuth_diff"] = np.minimum(df["azimuth_diff"], 360.0 - df["azimuth_diff"])
    df["antenna_gain"] = np.exp(-(df["azimuth_diff"] ** 2) / (2 * 30.0 ** 2))

    # 遮挡代理特征：只依赖几何，仿真/实测两张表都能算，定义完全一致
    df["blocked_feature"] = np.exp(-df["distance"] / 150.0) * 0.5

    return df
