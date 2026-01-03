# 异常值检测(IQR)
import pandas as pd
df=pd.read_csv("train.csv")
num=df.select_dtypes(include=[float,int]).columns.tolist()
with open("2a.txt","w",encoding="utf-8") as f:
    f.write("outlier detect\n")
    for k in num:
        q1=df[k].quantile(0.25); q3=df[k].quantile(0.75); i=q3-q1
        lo=q1-1.5*i; hi=q3+1.5*i
        n=((df[k]<lo)|(df[k]>hi)).sum()
        f.write(f"{k}:{n}\n")
print("done")