import os, glob
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

CLEAN_DIR = (Path(__file__).resolve().parent / "clean")
OUT_DIR = (Path(__file__).resolve().parent / "outputs")
RANDOM_STATE = 42
os.makedirs(OUT_DIR, exist_ok=True)

csvs = sorted(glob.glob(os.path.join(CLEAN_DIR, "*.csv")))
if not csvs:
    print("未在 'clean' 目录找到任何 csv 文件。")
for f in csvs:
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"读取失败：{f}，跳过。错误：{e}")
        continue

    subject = os.path.splitext(os.path.basename(f))[0]
    print(f"\n处理学科：{subject}，行数={len(df)}，列数={len(df.columns)}")

    # 目标 y = 第一列（排名），保证为数值
    y = pd.to_numeric(df.iloc[:, 0], errors='coerce')  # 第一列为排名
    # 学校列：优先 Institutions，否则回退到第一个 object 类型列
    if 'Institutions' in df.columns:
        names = df['Institutions'].astype(str)
    else:
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        names = df[obj_cols[0]].astype(str) if obj_cols else df.iloc[:, 1].astype(str)

    # 特征 X：去掉第一列（排名）和 Institutions（若存在）
    drop_cols = [df.columns[0]]
    if 'Institutions' in df.columns:
        drop_cols.append('Institutions')
    X = df.drop(columns=drop_cols, errors='ignore').copy()

    # 简单处理特征：数值保持，非数值做 one-hot（pd.get_dummies），缺失填 0（数值）或 0/0.0（dummy）
    # 先把所有列转换：对 object/非数值做 get_dummies
    nonnum = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if nonnum:
        X = pd.get_dummies(X, columns=nonnum, dummy_na=True, drop_first=True)
    # 将剩余的 NaN 用列中位数或 0 填补（这里统一用列中位数，若全为 NaN 用 0）
    for c in X.columns:
        if X[c].isna().any():
            col_med = X[c].median()
            X[c].fillna(col_med if not np.isnan(col_med) else 0, inplace=True)
    # 再确保 X 全是数值
    X = X.astype(float)

    # 丢掉目标缺失行（目标是未知则不能训练）
    mask = y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    names = names.loc[mask].reset_index(drop=True)
    if len(y) < 3:
        print(f"  有效样本太少（{len(y)}），跳过。")
        continue

    # 划分：先切出 20% 测试，再从剩余 80% 切出 25% 作为验证（即总量的 20%）
    X_temp, X_test, y_temp, y_test, names_temp, names_test = train_test_split(
        X, y, names, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )
    X_train, X_val, y_train, y_val, names_train, names_val = train_test_split(
        X_temp, y_temp, names_temp, test_size=0.25, random_state=RANDOM_STATE, shuffle=True
    )

    # 简单的验证：尝试两种 max_depth（简单且快速），用验证集选最优
    best_depth = None
    best_mae = 1e9
    for depth in (5, None):
        model = RandomForestRegressor(n_estimators=200, max_depth=depth, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)
        p = model.predict(X_val)
        m = mean_absolute_error(y_val, p)
        if m < best_mae:
            best_mae = m
            best_depth = depth

    # 用 train+val 训练最终模型
    X_comb = pd.concat([X_train, X_val], ignore_index=True)
    y_comb = pd.concat([y_train, y_val], ignore_index=True)
    final = RandomForestRegressor(n_estimators=300, max_depth=best_depth, random_state=RANDOM_STATE, n_jobs=-1)
    final.fit(X_comb, y_comb)

    pred_raw = final.predict(X_test)
    pred_round = np.rint(pred_raw).astype(int)
    mae = mean_absolute_error(y_test, pred_raw)

    out = pd.DataFrame({
        'Institutions': names_test,
        'true_rank': y_test.values,
        'pred_rank': pred_round,
        'pred_rank_raw': pred_raw
    })
    out = out.sort_values('true_rank').reset_index(drop=True)

    out_path = os.path.join(OUT_DIR, f"{subject}_test_predictions.csv")
    out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(out[['Institutions', 'true_rank', 'pred_rank']])
    print(f"MAE (test) = {mae:.4f}  -> 已保存: {out_path}")
