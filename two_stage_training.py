r"""
two_stage_training.py
=====================
步骤 3/4：XGBoost 两阶段训练（合成数据预训练 → 真实数据微调）+ 误差分析。

流程：
  阶段 1  在合成数据上训练，并用留出的合成基站早停
  阶段 2  以阶段 1 的 Booster 为**起点**，用更小的学习率在真实数据上继续训练
  输出    预测 vs 真实、误差分布、两阶段对比三张图；保存模型与 scaler

为什么阶段 2 必须显式传 xgb_model
--------------------------------
xgboost 的 sklearn 接口重复调用 fit() 会**丢弃已有树、从头重训**，官方文档原文：

    Note that calling fit() multiple times will cause the model object to be
    re-fit from scratch. To resume training from a previous checkpoint,
    explicitly pass xgb_model argument.

所以"预训练 + 微调"要成立，阶段 2 必须把阶段 1 的 Booster 通过 `xgb_model=`
传进去（training continuation）。本文件在训练结束后会**程序化校验**这一点：
把两个 Booster 的树序列 dump 出来，确认阶段 1 的树是阶段 2 树序列的前缀。
不是嘴上说续训，是能验证的续训。

两阶段同源的三条约束
--------------------
1. 特征：共用 features.py 里的 engineer_features —— 特征空间必须完全一致；
2. 标准化：共用同一个 scaler，且**只在合成数据的训练集上 fit**，
   真实数据的统计量从不参与，避免跨阶段统计量泄漏；
3. 超参：阶段 2 只降学习率（0.05 → 0.01）与追加轮数，其余保持一致 ——
   微调用小学习率是为了只在真实数据上做小幅修正，不把阶段 1 学到的
   距离—损耗、遮挡—衰减先验冲掉。

划分：按 station 分组（GroupShuffleSplit）
-----------------------------------------
同一基站的采样点在空间上高度相关，随机划分会让训练集与验证/测试集共享基站，
指标虚高。赛题真正考察的是**新基站的泛化能力**，所以按 cell_id 分组划分：
一个基站的全部采样点只落在 train / val / test 其中之一。
早停用 val，最终报出的 MAE 只在 test 上算一次（test 全程不参与任何选择）。

依赖：xgboost, scikit-learn, pandas, numpy, matplotlib, joblib
用法：
    set UCUP_DATA_ROOT=D:\path\to\TrainingData.26UCupSummer
    python two_stage_training.py
"""

import inspect
import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from features import engineer_features, get_feature_cols

import matplotlib
matplotlib.use("Agg")  # 无界面环境也能出图
import matplotlib.pyplot as plt  # noqa: E402

# ==================== 配置 ====================
DATA_ROOT = os.environ.get("UCUP_DATA_ROOT", "data/TrainingData.26UCupSummer")
SIM_DATA_PATH = os.environ.get(
    "UCUP_SIM_DATA", "outputs/simulation_data/simulation_data.csv"
)
OUTPUT_DIR = os.environ.get("UCUP_OUTPUT_DIR_2S", "outputs/two_stage_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GROUP_COL = "cell_id"   # 划分分组键：同一基站不跨集合
SEED = 42

# 阶段 1：合成数据上正常学习率训练
STAGE1_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=-1,
)

# 阶段 2：真实数据上小学习率微调；n_estimators 是**追加**的轮数，不是总轮数
STAGE2_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=-1,
)

EARLY_STOPPING_ROUNDS = 30

# xgboost >= 2.0 把 early_stopping_rounds 从 fit() 移到了构造函数；
# 这里探测一次，让新旧版本都能**真正**开启早停。
# （如果只在 fit() 上盲试再 except 兜底，在新版上会静默退化成"不早停"，跑满全部轮数。）
_FIT_TAKES_EARLY_STOP = (
    "early_stopping_rounds" in inspect.signature(xgb.XGBRegressor.fit).parameters
)


# ==================== 加载数据 ====================
def load_real_data():
    """加载所有基站的实测数据。

    只要目录里有 ep.json + train_signal.csv 就当训练基站参与，
    不再靠目录名前缀判断（前缀规则会误伤 cell_10x 这种编号）。
    """
    print("加载真实数据...")
    if not os.path.isdir(DATA_ROOT):
        raise SystemExit(f"数据目录不存在: {DATA_ROOT}\n请设置环境变量 UCUP_DATA_ROOT")

    all_data = []
    for name in sorted(os.listdir(DATA_ROOT)):
        cell_path = os.path.join(DATA_ROOT, name)
        if not os.path.isdir(cell_path):
            continue
        ep_path = os.path.join(cell_path, "ep.json")
        signal_path = os.path.join(cell_path, "train_signal.csv")
        if not (os.path.exists(ep_path) and os.path.exists(signal_path)):
            continue

        with open(ep_path, "r") as f:
            ep = json.load(f)

        df = pd.read_csv(signal_path)
        df["cell_id"] = name
        df["antenna_height"] = ep.get("height", 30)
        df["azimuth"] = ep.get("azimuth", 0)
        df["downtilt"] = ep.get("downtilt", 5)
        all_data.append(df)

    if not all_data:
        print("没有找到真实数据")
        return None
    df = pd.concat(all_data, ignore_index=True)
    print(f"   真实样本: {len(df):,}, 基站: {df['cell_id'].nunique()}")
    return df


def load_simulation_data():
    """加载合成数据。"""
    print("加载合成数据...")
    if not os.path.exists(SIM_DATA_PATH):
        print(f"   合成数据不存在: {SIM_DATA_PATH}")
        return None
    df = pd.read_csv(SIM_DATA_PATH)
    if GROUP_COL not in df.columns:
        raise SystemExit(f"合成数据缺少 {GROUP_COL} 列，无法按基站分组划分: {SIM_DATA_PATH}")
    print(f"   合成样本: {len(df):,}, 基站: {df[GROUP_COL].nunique()}")
    print(f"   RSRP 范围: {df['rsrp'].min():.1f} ~ {df['rsrp'].max():.1f} dBm")
    return df


# ==================== 划分与早停 ====================
def group_holdout(df, test_size, seed):
    """按 GROUP_COL 分组切出一份 holdout，返回 (保留位置索引, holdout 位置索引)。

    用 GroupShuffleSplit 而不是 train_test_split：分组键相同的样本
    （同一基站的全部采样点）整体进同一侧。
    """
    pos = np.arange(len(df))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    keep, hold = next(splitter.split(pos, groups=df[GROUP_COL].values))
    return pos[keep], pos[hold]


def report_split(label, df, idx):
    sub = df.iloc[idx]
    print(f"   {label:<10} 样本 {len(sub):>9,}   基站 {sub[GROUP_COL].nunique():>3}")


def build_model(params, early_stopping_rounds=None):
    """构造 XGBRegressor；早停按 xgboost 版本决定放构造函数还是 fit()。"""
    params = dict(params)
    if early_stopping_rounds is not None and not _FIT_TAKES_EARLY_STOP:
        params["early_stopping_rounds"] = early_stopping_rounds
    return xgb.XGBRegressor(**params)


def fit_model(model, X_tr, y_tr, X_ev, y_ev, xgb_model=None):
    """带早停的 fit；xgb_model 非空时是 training continuation，不是重训。"""
    kwargs = {"eval_set": [(X_tr, y_tr), (X_ev, y_ev)], "verbose": False}
    if _FIT_TAKES_EARLY_STOP:
        kwargs["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    if xgb_model is not None:
        kwargs["xgb_model"] = xgb_model
    model.fit(X_tr, y_tr, **kwargs)
    return model


def dump_trees(booster):
    """取出 Booster 里的树序列，用于验证"续训"是否真的成立。"""
    raw = json.loads(booster.save_raw(raw_format="json"))
    return raw["learner"]["gradient_booster"]["model"]["trees"]


def verify_continuation(booster_stage1, booster_stage2):
    """确认阶段 1 的树构成阶段 2 树序列的前缀。

    成立 => 阶段 2 是在阶段 1 的模型上继续加树（真正的预训练 + 微调）；
    不成立 => 说明阶段 2 实际上是从头重训，xgb_model 没生效。
    """
    try:
        t1 = dump_trees(booster_stage1)
        t2 = dump_trees(booster_stage2)
    except Exception as e:  # 老版本可能不支持 json 导出
        print(f"   [跳过校验] 无法导出树结构: {e}")
        return None
    if len(t2) < len(t1):
        return False
    return json.dumps(t2[: len(t1)], sort_keys=True) == json.dumps(t1, sort_keys=True)


# ==================== 两阶段训练 ====================
def train_two_stage(sim_df, real_df):
    print("\n" + "=" * 60)
    print("  两阶段训练")
    print("=" * 60)

    feature_cols = get_feature_cols()
    print(f"\n构造特征（{len(feature_cols)} 维，两阶段共用同一实现）...")
    sim_df = engineer_features(sim_df)
    real_df = engineer_features(real_df)

    # ---- 划分：按基站分组 ----
    print("\n按基站分组划分（GroupShuffleSplit, groups=cell_id）:")
    sim_tr, sim_val = group_holdout(sim_df, test_size=0.2, seed=SEED)
    real_rest, real_test = group_holdout(real_df, test_size=0.2, seed=SEED)
    rel_tr, rel_val = group_holdout(
        real_df.iloc[real_rest].reset_index(drop=True), test_size=0.2, seed=SEED + 1
    )
    real_tr, real_val = real_rest[rel_tr], real_rest[rel_val]

    report_split("合成训练", sim_df, sim_tr)
    report_split("合成早停", sim_df, sim_val)
    report_split("真实训练", real_df, real_tr)
    report_split("真实早停", real_df, real_val)
    report_split("真实测试", real_df, real_test)

    # ---- 标准化：只用合成训练集 fit，两个阶段共用同一个 scaler ----
    scaler = StandardScaler().fit(sim_df.iloc[sim_tr][feature_cols].values)

    def pack(idx, df):
        X = scaler.transform(df.iloc[idx][feature_cols].values)
        y = df.iloc[idx]["rsrp"].values
        return X, y

    X_sim_tr, y_sim_tr = pack(sim_tr, sim_df)
    X_sim_val, y_sim_val = pack(sim_val, sim_df)
    X_real_tr, y_real_tr = pack(real_tr, real_df)
    X_real_val, y_real_val = pack(real_val, real_df)
    X_real_test, y_real_test = pack(real_test, real_df)

    def acc(y_true, y_pred):
        return (
            mean_absolute_error(y_true, y_pred),
            float(np.sqrt(mean_squared_error(y_true, y_pred))),
        )

    # ===== 阶段 1：合成数据预训练 =====
    print("\n" + "=" * 60)
    print("  阶段 1: 合成数据预训练")
    print("=" * 60)

    stage1 = build_model(STAGE1_PARAMS, EARLY_STOPPING_ROUNDS)
    fit_model(stage1, X_sim_tr, y_sim_tr, X_sim_val, y_sim_val)
    booster_stage1 = stage1.get_booster()
    trees_stage1 = len(dump_trees(booster_stage1))

    mae_sim, rmse_sim = acc(y_sim_val, stage1.predict(X_sim_val))
    print(f"\n合成早停集: MAE {mae_sim:.2f} dB / RMSE {rmse_sim:.2f} dB"
          f"   （早停后保留 {trees_stage1} 棵树）")

    pred_pre = stage1.predict(X_real_test)
    mae_pre, rmse_pre = acc(y_real_test, pred_pre)
    print("真实测试集（仅合成数据训练出来的模型）:")
    print(f"   MAE {mae_pre:.2f} dB / RMSE {rmse_pre:.2f} dB")
    print("   ↑ 这个数就是 sim-to-real gap 的量化：仿真器里学到的规律换到实测数据上不够用，"
          "\n     所以还需要阶段 2 在真实数据上继续训练。")

    # ===== 阶段 2：真实数据微调（真正续训） =====
    print("\n" + "=" * 60)
    print("  阶段 2: 真实数据微调（xgb_model 续训，学习率 0.01）")
    print("=" * 60)

    stage2 = build_model(STAGE2_PARAMS, EARLY_STOPPING_ROUNDS)
    fit_model(stage2, X_real_tr, y_real_tr, X_real_val, y_real_val,
              xgb_model=booster_stage1)
    booster_stage2 = stage2.get_booster()
    trees_stage2 = len(dump_trees(booster_stage2))

    continued = verify_continuation(booster_stage1, booster_stage2)
    if continued is True:
        print(f"\n续训校验通过: 阶段 1 的 {trees_stage1} 棵树是阶段 2 树序列的前缀，"
              f"阶段 2 追加至 {trees_stage2} 棵 —— 确实是 training continuation，不是重训。")
    elif continued is False:
        raise SystemExit(
            "续训校验失败：阶段 2 的树序列不是阶段 1 的超集，说明 xgb_model 没有生效。"
        )
    else:
        print(f"\n阶段 1 {trees_stage1} 棵树 → 阶段 2 {trees_stage2} 棵。")

    pred_final = stage2.predict(X_real_test)
    mae_real, rmse_real = acc(y_real_test, pred_final)
    print("\n真实测试集（微调后的模型）:")
    print(f"   MAE {mae_real:.2f} dB / RMSE {rmse_real:.2f} dB")
    print(f"   相对阶段 1: {mae_pre:.2f} → {mae_real:.2f} dB "
          f"（改善 {mae_pre - mae_real:.2f} dB）")

    results = {
        "mae_sim": mae_sim,
        "rmse_sim": rmse_sim,
        "mae_real_pretrain": mae_pre,
        "rmse_real_pretrain": rmse_pre,
        "mae_real": mae_real,
        "rmse_real": rmse_real,
        "improvement": mae_pre - mae_real,
        "trees_stage1": trees_stage1,
        "trees_stage2": trees_stage2,
        "continued_ok": continued,
    }
    return stage1, stage2, scaler, feature_cols, results, (X_real_test, y_real_test)


# ==================== 可视化 ====================
def visualize_results(model, X_test, y_test, results):
    print("\n" + "=" * 60)
    print("  可视化结果")
    print("=" * 60)
    # 注意：这里只用**留出的真实测试基站**，不含任何训练/早停样本
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    errors = y_pred - y_test

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.3, s=5)
    lo, hi = float(y_test.min()), float(y_test.max())
    ax1.plot([lo, hi], [lo, hi], "r--")
    ax1.set_xlabel("真实 RSRP (dBm)")
    ax1.set_ylabel("预测 RSRP (dBm)")
    ax1.set_title(f"预测 vs 真实（留出基站）\nMAE: {mae:.2f} dB")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(errors, bins=50, alpha=0.7, color="steelblue")
    ax2.axvline(0, color="red", linestyle="--")
    ax2.set_xlabel("误差 (dB)")
    ax2.set_ylabel("频数")
    ax2.set_title(f"误差分布\nstd: {errors.std():.2f} dB")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    metrics = ["MAE", "RMSE"]
    pre = [results["mae_real_pretrain"], results["rmse_real_pretrain"]]
    fin = [results["mae_real"], results["rmse_real"]]
    x = np.arange(len(metrics))
    width = 0.35
    ax3.bar(x - width / 2, pre, width, label="仅合成数据训练", color="tab:blue", alpha=0.75)
    ax3.bar(x + width / 2, fin, width, label="真实数据微调后", color="tab:green", alpha=0.75)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.set_ylabel("误差 (dB)")
    ax3.set_title("两阶段对比（同一留出测试集）")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "two_stage_results.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"   可视化保存: {output_path}")


# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("  两阶段训练: XGBoost")
    print("=" * 60)

    sim_df = load_simulation_data()
    if sim_df is None:
        return
    real_df = load_real_data()
    if real_df is None:
        return

    stage1, stage2, scaler, feature_cols, results, (X_test, y_test) = train_two_stage(
        sim_df, real_df
    )
    visualize_results(stage2, X_test, y_test, results)

    print("\n" + "=" * 60)
    print("  保存模型")
    print("=" * 60)
    artifact = {"model": stage2, "scaler": scaler, "feature_cols": feature_cols}
    model_path = os.path.join(OUTPUT_DIR, "two_stage_model.pkl")
    joblib.dump(artifact, model_path)
    print(f"   微调后模型（提交用）: {model_path}")

    # 阶段 1 的模型单独留一份，方便复现"仅合成数据"的那一行指标
    stage1_path = os.path.join(OUTPUT_DIR, "stage1_sim_model.pkl")
    joblib.dump({"model": stage1, "scaler": scaler, "feature_cols": feature_cols},
                stage1_path)
    print(f"   阶段 1 模型（仅留档）: {stage1_path}")

    print("\n" + "=" * 60)
    print("  最终结果（真实留出测试基站）")
    print("=" * 60)
    print(f"   仅合成数据训练 MAE: {results['mae_real_pretrain']:.2f} dB")
    print(f"   真实数据微调后 MAE: {results['mae_real']:.2f} dB")
    print(f"   改善: {results['improvement']:.2f} dB")
    print("   ⚠️ 本地自测口径，非赛事榜单分数")
    print("=" * 60)


if __name__ == "__main__":
    main()
