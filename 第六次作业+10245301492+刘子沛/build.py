from __future__ import annotations

from pathlib import Path
import pandas as pd


INPUT_DIR = (Path(__file__).resolve().parent / "clean")
OUTPUT_CSV = (Path(__file__).resolve().parent / "build.csv")


def list_subject_files(root: Path) -> list[Path]:
    files = sorted(root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {root}")
    return files


def read_subject(csv_path: Path) -> pd.DataFrame:
    """
    读取单个学科表，返回三列：['university', 'subject', 'rank']。
    - rank 取文件的第一列（清洗后的名次列）
    - university 取 'Institutions'
    - subject 用文件名（去掉扩展名）
    """
    df = pd.read_csv(csv_path, dtype=str)
    rank_col = df.columns[0]               # 第一列为名次
    inst_col = "Institutions"              # 清洗后固定为该列名

    out = pd.DataFrame({
        "university": df[inst_col].astype(str).str.strip(),
        "rank": pd.to_numeric(df[rank_col], errors="coerce"),
    }).dropna(subset=["university", "rank"])

    out["rank"] = out["rank"].astype(int)
    out["subject"] = csv_path.stem.strip()
    return out[["university", "subject", "rank"]]


def build_matrix(input_dir: Path) -> pd.DataFrame:
    """合并所有学科并透视为 高校×学科 的名次矩阵。"""
    frames = [read_subject(p) for p in list_subject_files(input_dir)]
    long_df = pd.concat(frames, ignore_index=True)

    # 若同一高校在同一学科出现多次，取最好名次（最小值）
    pivot = long_df.pivot_table(
        index="university",
        columns="subject",
        values="rank",
        aggfunc="min",
    )

    # 按行、列排序，输出更稳定
    return pivot.sort_index().sort_index(axis=1)


def main() -> None:
    pivot = build_matrix(INPUT_DIR)
    pivot.to_csv(OUTPUT_CSV, encoding="utf-8-sig")
    print(f"Done: {OUTPUT_CSV}  shape={pivot.shape}")


if __name__ == "__main__":
    main()
