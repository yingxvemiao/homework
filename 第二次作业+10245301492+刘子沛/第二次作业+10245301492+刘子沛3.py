# 相关性矩阵+去冗余
import pandas as pd
df=pd.read_csv("train.csv")
cols=[k for k in df.columns if pd.api.types.is_numeric_dtype(df[k])]
cor=df[cols].corr().abs()
order=df[cols].var().sort_values(ascending=False).index.tolist()
keep=[]
for k in order:
    ok=True
    for j in keep:
        if cor.loc[k,j]>0.4: ok=False; break
    if ok: keep.append(k)
cor_keep=cor.loc[keep,keep].round(3)
cor_keep.to_csv("3a.csv")
print("done")