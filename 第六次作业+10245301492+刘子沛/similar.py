from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# === 1. 读取并预处理数据 ===
path = (Path(__file__).resolve().parent / "build.csv")  # 数据路径，可自行修改
df = pd.read_csv(path)
universities = df.iloc[:, 0]  # 第一列为学校名
X = df.iloc[:, 1:].fillna(df.max() * 2)  # 其余列为各学科排名

# === 2. 数据标准化 ===
X_scaled = StandardScaler().fit_transform(X)

# === 3. 聚类（KMeans 或其他算法） ===
kmeans = KMeans(n_clusters=100, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
df["Cluster"] = labels
# === 4. 查找与华东师范大学同簇学校 ===
target = "EAST CHINA NORMAL UNIVERSITY"
if target in universities.values:
    target_cluster = df.loc[universities == target, "Cluster"].values[0]
    print(target_cluster)
    name_col = df.columns[0]
    similar_schools = df[df["Cluster"] == target_cluster][name_col].values
    print(f"与 {target} 聚类相似的高校：")
    print(similar_schools)
else:
    print(f"未找到 {target}")

# === 5. 计算相似度（补充分析） ===
target_vec = X_scaled[universities.str.upper() == target.upper()][0].reshape(1, -1)
similarity = cosine_similarity(target_vec, X_scaled)[0]
df["Similarity"] = similarity

print("\n最相似的高校（按相似度降序）：")
print(df.sort_values("Similarity", ascending=False).head(50)[[name_col, "Similarity"]])
