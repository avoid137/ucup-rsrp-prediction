r"""
smoke_test.py
=============
合成数据冒烟测试：不碰赛题数据，用假数据把「训练 → 推理」全流程跑通，
并**程序化断言**几条容易口头声称、但实际做不到的行为。

它验证的是**逻辑**，不是成绩（假数据上的 MAE 没有任何意义）。

检查项
------
1. 分组划分：train / val / test 三个集合的 `cell_id` **两两不相交**
2. 特征同源：`engineer_features` 对"带干扰列的表"与"干净表"给出**完全一致**的特征值
3. 续训成立：阶段 1 的树序列是阶段 2 树序列的**前缀**（真 training continuation，不是重训）
4. 早停生效：阶段 1 保留的树数 **< n_estimators**（否则说明早停被静默关掉了）
5. 分组划分真的按组切：同一 cell_id 的全部样本只出现在一侧
6. 提交产物：`output.zip` 生成，且每个测试基站一个 CSV

用法：
    python smoke_test.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import features  # noqa: E402,F401  （确认模块可正常 import）
import two_stage_training as T  # noqa: E402

N_TRAIN_CELLS = 10
N_TEST_CELLS = 3
POINTS_PER_CELL = 400
SIM_POINTS_PER_CELL = 300


def make_fake_dataset(root):
    """造一份结构同赛题、但内容全为随机的假数据集。"""
    rng = np.random.default_rng(7)
    data_root = os.path.join(root, "TrainingData")
    out_root = os.path.join(root, "outputs")
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(os.path.join(out_root, "simulation_data"), exist_ok=True)

    def geometry(n, rng):
        r = rng.uniform(20, 500, n)
        th = rng.uniform(0, 360, n)
        return r * np.cos(np.radians(th)), r * np.sin(np.radians(th))

    # ---- 训练基站：ep.json + train_signal.csv ----
    for i in range(N_TRAIN_CELLS):
        name = f"cell_{i:03d}"
        d = os.path.join(data_root, name)
        os.makedirs(d, exist_ok=True)
        height, az = 25 + i, float((i * 37) % 360)
        with open(os.path.join(d, "ep.json"), "w") as f:
            json.dump({"height": height, "azimuth": az, "downtilt": 6.0}, f)

        x, y = geometry(POINTS_PER_CELL, rng)
        pl = 20 * np.log10(np.sqrt(x ** 2 + y ** 2) + 1) + 20 * np.log10(3.5e9) - 147.55
        # 基站级偏置：这样"同一基站跨集合"才会让指标虚高，检查项 1/5 才有意义
        rsrp = np.clip(46 - pl - i * 0.8 + rng.normal(0, 3, POINTS_PER_CELL), -130, -40)
        pd.DataFrame({"point_id": np.arange(POINTS_PER_CELL), "x": x, "y": y,
                      "rsrp": rsrp}).to_csv(os.path.join(d, "train_signal.csv"),
                                            index=False)

    # ---- 测试基站：ep.json + test_points.csv ----
    for j in range(N_TEST_CELLS):
        name = f"cell_{100 + j:03d}"
        d = os.path.join(data_root, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "ep.json"), "w") as f:
            json.dump({"height": 30, "azimuth": 90.0, "downtilt": 5.0}, f)
        x, y = geometry(50, rng)
        pd.DataFrame({"point_id": np.arange(50), "x": x, "y": y}).to_csv(
            os.path.join(d, "test_points.csv"), index=False)

    # ---- 仿真数据：刻意与"实测"存在系统性偏差，模拟 sim-to-real gap ----
    rows = []
    for i in range(N_TRAIN_CELLS):
        x, y = geometry(SIM_POINTS_PER_CELL, rng)
        pl = 20 * np.log10(np.sqrt(x ** 2 + y ** 2) + 1) + 20 * np.log10(3.5e9) - 147.55
        rows.append(pd.DataFrame({
            "cell_id": f"cell_{i:03d}", "x": x, "y": y,
            "rsrp": np.clip(46 - pl + 14 + rng.normal(0, 2, SIM_POINTS_PER_CELL),
                            -130, -40),
            "antenna_height": 25 + i, "azimuth": float((i * 37) % 360),
            "downtilt": 6.0}))
    sim = pd.concat(rows, ignore_index=True)
    sim_path = os.path.join(out_root, "simulation_data", "simulation_data.csv")
    sim.to_csv(sim_path, index=False)

    return data_root, out_root, sim_path


def run(script, env_extra):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", **env_extra}
    p = subprocess.run([sys.executable, os.path.join(HERE, script)],
                       cwd=HERE, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    if p.returncode != 0:
        print(p.stdout)
        print("STDERR:", p.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"{script} 退出码 {p.returncode}")
    return p.stdout


# ==================== 检查项 ====================
def check_group_split():
    """1 + 5：三集合的 cell_id 两两不相交。"""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"cell_id": [f"c{i:02d}" for i in range(20) for _ in range(5)],
                       "v": rng.normal(size=100)})
    rest, test = T.group_holdout(df, 0.2, 42)
    rel_tr, rel_val = T.group_holdout(
        df.iloc[rest].reset_index(drop=True), 0.2, 43)
    tr, val = rest[rel_tr], rest[rel_val]
    s_tr, s_val, s_te = (set(df.iloc[i]["cell_id"]) for i in (tr, val, test))
    assert not (s_tr & s_val), f"train 与 val 共享基站: {s_tr & s_val}"
    assert not (s_tr & s_te), f"train 与 test 共享基站: {s_tr & s_te}"
    assert not (s_val & s_te), f"val 与 test 共享基站: {s_val & s_te}"
    return f"三集合基站两两不相交（{len(s_tr)}/{len(s_val)}/{len(s_te)} 个）"


def check_feature_same_source():
    """2：同名不同义的列不得沿用，两张表必须得到同一套特征值。"""
    base = {"x": [100.0, -100.0, 0.0, 30.0], "y": [0.0, 0.0, 80.0, -40.0],
            "antenna_height": [30.0] * 4, "azimuth": [0.0, 180.0, 90.0, 270.0],
            "downtilt": [5.0] * 4}
    polluted = {**base,                       # 仿真表里带着"看上去能用"的同名列
                "distance": [999.0] * 4, "angle": [-90.0] * 4,
                "is_blocked": [0, 1, 0, 1]}
    a = T.engineer_features(pd.DataFrame(polluted))
    b = T.engineer_features(pd.DataFrame(base))
    cols = T.get_feature_cols()
    assert len(cols) == 17, f"特征维度应为 17，实际 {len(cols)}"
    assert np.allclose(a[cols].values, b[cols].values), "两张表特征值不一致"
    assert np.allclose(a["distance"].values, [100, 100, 80, 50]), "distance 未按 x,y 现算"
    assert a["angle"].min() >= 0 and a["angle"].max() < 360, "angle 未归一到 [0,360)"
    return "两表特征完全一致，distance 现算、angle 归一化"


def check_continuation_and_early_stop(stdout):
    """3 + 4：真续训 + 早停生效。"""
    assert "续训校验通过" in stdout, "阶段 2 不是续训（树序列前缀校验未通过）"
    import re
    m = re.search(r"早停后保留 (\d+) 棵树", stdout)
    assert m, "未打印早停后保留的树数"
    kept = int(m.group(1))
    assert kept < T.STAGE1_PARAMS["n_estimators"], \
        f"树数 {kept} 达到 n_estimators，早停没生效"
    return f"续训前缀校验通过；早停生效（保留 {kept} < {T.STAGE1_PARAMS['n_estimators']} 轮）"


def check_submission(out_root):
    """6：提交包生成且每基站一个 CSV。"""
    zip_path = os.path.join(out_root, "submission", "output.zip")
    assert os.path.exists(zip_path), f"未生成 {zip_path}"
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        cols = pd.read_csv(zf.open(names[0])).columns.tolist()
    assert len(names) == N_TEST_CELLS, f"应有 {N_TEST_CELLS} 个 CSV，实际 {len(names)}"
    assert cols == ["point_id", "rsrp_pred"], f"CSV 列名不符: {cols}"
    return f"output.zip 含 {len(names)} 个基站，列名 {cols}"


def main():
    tmp = tempfile.mkdtemp(prefix="ucup_smoke_")
    try:
        data_root, out_root, sim_path = make_fake_dataset(tmp)
        print(f"假数据集: {data_root}\n")

        env_extra = {
            "UCUP_DATA_ROOT": data_root,
            "UCUP_SIM_DATA": sim_path,
            "UCUP_OUTPUT_DIR_2S": os.path.join(out_root, "two_stage_results"),
            "UCUP_MODEL_PATH": os.path.join(out_root, "two_stage_results",
                                            "two_stage_model.pkl"),
            "UCUP_SUBMISSION_DIR": os.path.join(out_root, "submission"),
        }

        print("=" * 60)
        print("  1/2  运行 two_stage_training.py")
        print("=" * 60)
        train_out = run("two_stage_training.py", env_extra)

        print("=" * 60)
        print("  2/2  运行 generate_submission.py")
        print("=" * 60)
        run("generate_submission.py", env_extra)

        checks = [
            ("分组划分不跨基站", check_group_split),
            ("特征两阶段同源", check_feature_same_source),
            ("阶段 2 是真续训 + 早停生效",
             lambda: check_continuation_and_early_stop(train_out)),
            ("提交包结构正确", lambda: check_submission(out_root)),
        ]

        print("\n" + "=" * 60)
        print("  检查结果")
        print("=" * 60)
        failed = 0
        for name, fn in checks:
            try:
                print(f"  [PASS] {name}: {fn()}")
            except AssertionError as e:
                failed += 1
                print(f"  [FAIL] {name}: {e}")
        print("=" * 60)
        print(f"  {len(checks) - failed}/{len(checks)} 通过")
        return 1 if failed else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
