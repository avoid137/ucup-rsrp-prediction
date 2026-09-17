r"""
generate_submission.py
======================
步骤 4/4：加载微调后的模型，对测试基站逐点预测 RSRP，打包成 output.zip 提交。

输出结构：
    outputs/submission/output/<cell_id>.csv   # 每个测试点一行: point_id, rsrp_pred
    outputs/submission/output.zip             # 把所有 csv 打包，供平台提交

一个关键点：这里直接 import features.py 里的 engineer_features —— 与训练脚本
用的是**同一份特征实现**。最初版本把这段代码在训练/推理两个文件里各写了一份，
只要有一边改动就会特征漂移，而特征漂移在这类比赛里是无声的：不报错，
分数直接掉一半。抽成公共模块后，这种错在结构上不可能发生。

依赖：pandas, numpy, joblib, 以及训练时的 xgboost（joblib 反序列化需要）
用法：
    set UCUP_DATA_ROOT=D:\path\to\TrainingData.26UCupSummer
    python generate_submission.py
"""

import json
import os
import zipfile

import joblib
import numpy as np
import pandas as pd

from features import engineer_features

# ==================== 配置 ====================
DATA_ROOT = os.environ.get("UCUP_DATA_ROOT", "data/TrainingData.26UCupSummer")
MODEL_PATH = os.environ.get(
    "UCUP_MODEL_PATH", "outputs/two_stage_results/two_stage_model.pkl"
)
OUTPUT_DIR = os.environ.get("UCUP_SUBMISSION_DIR", "outputs/submission")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 预测值裁剪到合法物理范围（与仿真器的标签裁剪一致）
RSRP_MIN, RSRP_MAX = -130, -40


def predict_cell(cell_dir, model_data):
    """预测单个测试基站。"""
    ep_path = os.path.join(cell_dir, "ep.json")
    test_path = os.path.join(cell_dir, "test_points.csv")

    if not (os.path.exists(ep_path) and os.path.exists(test_path)):
        return None

    with open(ep_path, "r") as f:
        ep = json.load(f)

    df = pd.read_csv(test_path)
    df["antenna_height"] = ep.get("height", 30)
    df["azimuth"] = ep.get("azimuth", 0)
    df["downtilt"] = ep.get("downtilt", 5)

    # 与训练完全同一份特征实现（features.py）
    df = engineer_features(df)

    feature_cols = model_data["feature_cols"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"特征缺失 {missing}，训练与推理的特征定义不一致")

    X = df[feature_cols].values
    X_scaled = model_data["scaler"].transform(X)
    y_pred = np.clip(model_data["model"].predict(X_scaled), RSRP_MIN, RSRP_MAX)

    return pd.DataFrame({"point_id": df["point_id"], "rsrp_pred": y_pred})


def main():
    print("=" * 60)
    print("  生成提交文件")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"模型不存在: {MODEL_PATH}\n请先运行 two_stage_training.py")

    model_data = joblib.load(MODEL_PATH)
    print(f"模型加载成功: {MODEL_PATH}")

    # 测试基站的判定标准是"有 test_points.csv"，不靠目录名前缀
    # （前缀规则会把 cell_10x 这类训练基站误判成测试基站）
    cell_dirs = []
    for name in sorted(os.listdir(DATA_ROOT)):
        path = os.path.join(DATA_ROOT, name)
        if os.path.isdir(path) and os.path.exists(
            os.path.join(path, "test_points.csv")
        ):
            cell_dirs.append(path)
    print(f"找到 {len(cell_dirs)} 个测试基站")
    if not cell_dirs:
        raise SystemExit(
            f"在 {DATA_ROOT} 下没有找到含 test_points.csv 的目录，请检查 UCUP_DATA_ROOT"
        )

    output_sub_dir = os.path.join(OUTPUT_DIR, "output")
    os.makedirs(output_sub_dir, exist_ok=True)

    written = 0
    for i, cell_dir in enumerate(cell_dirs, 1):
        cell_name = os.path.basename(cell_dir)
        print(f"[{i}/{len(cell_dirs)}] {cell_name}")

        result = predict_cell(cell_dir, model_data)
        if result is None:
            print("   跳过（缺 ep.json 或 test_points.csv）")
            continue
        cell_id = cell_name.replace("cell_", "")
        result.to_csv(os.path.join(output_sub_dir, f"{cell_id}.csv"), index=False)
        written += 1
        print(f"   {cell_id}.csv ({len(result)} 行)")

    # 打包
    zip_path = os.path.join(OUTPUT_DIR, "output.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(output_sub_dir)):
            if f.endswith(".csv"):
                zf.write(os.path.join(output_sub_dir, f), f)

    print(f"\nZIP: {zip_path}  （含 {written} 个基站）")
    print("=" * 60)


if __name__ == "__main__":
    main()
