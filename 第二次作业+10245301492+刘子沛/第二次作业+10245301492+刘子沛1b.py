# 缺失值处理(KNN)
import os,pandas as pd,numpy as np
from sklearn.impute import KNNImputer
df=pd.read_csv("train.csv")
num=df.select_dtypes(include=[np.number]).columns.tolist()
cat=df.select_dtypes(exclude=[np.number]).columns.tolist()
imp=KNNImputer(n_neighbors=5)
df[num]=imp.fit_transform(df[num])
for k in cat:
    m=df[k].mode(dropna=True)
    val=m.iloc[0] if len(m)>0 else "Missing"
    df[k]=df[k].fillna(val)
out="trainKNN.csv"
df.to_csv(out,index=False)
print("done")