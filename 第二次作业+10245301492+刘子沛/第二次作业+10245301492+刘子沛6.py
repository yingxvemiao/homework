# 与price最相关的三个特征
import pandas as pd,numpy as np
df=pd.read_csv("train.csv")
col="price" if "price" in df.columns else ("SalePrice" if "SalePrice" in df.columns else None)
if col is None: raise SystemExit("no price")
y=df[col].astype(float)
num=[k for k in df.columns if pd.api.types.is_numeric_dtype(df[k]) and k!=col]
cat=[k for k in df.columns if not pd.api.types.is_numeric_dtype(df[k])]
s={}
if num:
    v=df[num+[col]].corr()[col].abs().drop(col)
    for k in v.index: s[k]=v.loc[k]
for k in cat:
    d=pd.get_dummies(df[k])
    m=0.0
    for j in d.columns:
        t=pd.concat([d[j],y],axis=1).corr().iloc[0,1]
        if pd.notna(t): m=max(m,abs(t))
    s[k]=m
top=sorted(s.items(),key=lambda x:x[1],reverse=True)[:3]
for k,v in top: print(k,round(v,3))