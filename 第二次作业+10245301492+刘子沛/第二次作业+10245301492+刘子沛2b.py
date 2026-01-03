# 异常值处理(均值替换)
import pandas as pd
df=pd.read_csv("train.csv")
num=df.select_dtypes(include=[float,int]).columns.tolist()
for k in num:
    s=df[k]
    q1=s.quantile(0.25); q3=s.quantile(0.75); i=q3-q1
    lo=q1-1.5*i; hi=q3+1.5*i
    m=s.mean()
    df[k]=s.mask((s<lo)|(s>hi),m)
df.to_csv("trainMean.csv",index=False)
print("done")