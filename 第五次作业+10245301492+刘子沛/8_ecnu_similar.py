import pandas as pd, numpy as np
from pathlib import Path

CSV = (Path(__file__).resolve().parent / "8_build.csv")
TARGET = "EAST CHINA NORMAL UNIVERSITY"
OUT = (Path(__file__).resolve().parent / "8_ecnu_similar.txt")

# 读表与转数字
df = pd.read_csv(CSV)
u, subs = df.columns[0], df.columns[1:]
df[subs] = df[subs].apply(pd.to_numeric, errors="coerce")

# 名次 -> [0,1] 得分；未上榜给极低分（很靠后）
m = df[subs].max(skipna=True)
den = (m - 1).replace(0, np.nan)
score = 1 - (df[subs] - 1).div(den, axis=1)
score = score.clip(0, 1).fillna(1 / (m + 1))
S = pd.concat([df[[u]], score], axis=1)

# 找目标（英文或中文名）
mask = S[u].str.contains(TARGET, case=False, na=False) | S[u].str.contains("华东师范", na=False)
if not mask.any():
    raise SystemExit(f"target not found: {TARGET}")

t = S.loc[mask, subs].iloc[0].to_numpy(float)
M = S.loc[~mask, subs].to_numpy(float)
names = S.loc[~mask, u].to_numpy()

# 余弦相似度
denom = np.linalg.norm(M, axis=1) * np.linalg.norm(t)
sims = (M @ t) / denom
ok = np.isfinite(sims)
names, sims = names[ok], sims[ok]

# 排序并输出到 txt
order = np.argsort(-sims)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(f"Similar to EAST CHINA NORMAL UNIVERSITY (N={len(order)})\n")
    for i, idx in enumerate(order, 1):
        f.write(f"{i:4d}. {names[idx]} | sim={sims[idx]:.4f}\n")

print(f"done -> {OUT}")