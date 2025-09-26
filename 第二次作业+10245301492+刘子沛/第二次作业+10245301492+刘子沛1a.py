# 缺失值的检测
import os
import pandas as pd
df=pd.read_csv("train.csv")
r,c=df.shape
mc=df.isna().sum()
mr=(df.isna().mean()*100).round(2)
tot=int(mc.sum())
rat=round(tot/(r*c)*100,2)
with open("1a.txt","w",encoding="utf-8") as f:
    f.write(f"shape:{r} x {c}\n\n")
    f.write("per col count:\n")
    for k,v in mc.sort_values(ascending=False).items(): f.write(f"{k}:{v}\n")
    f.write("\nper col rate(%):\n")
    for k,v in mr.sort_values(ascending=False).items(): f.write(f"{k}:{v}\n")
    f.write(f"\ntotal:{tot}\n")
    f.write(f"total rate(%):{rat}\n")
print("done")
