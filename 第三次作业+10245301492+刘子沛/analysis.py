# -*- coding: utf-8 -*-
"""
从 data/ 目录中的每个学科 .xlsx 文件里，查找
  "EAST CHINA NORMAL UNIVERSITY" 所在行，
将该行（加上学科名）汇总输出到 analysis.xlsx。

依赖：pandas、openpyxl
    pip install pandas openpyxl
"""

import os
import re
import sys
import pandas as pd
import numpy as np

INPUT_DIR = r"D:\ECNU\数据科学导论\第三次作业\data"
OUTPUT_XLSX = r"D:\ECNU\数据科学导论\第三次作业\analysis.xlsx"

TARGET = "EAST CHINA NORMAL UNIVERSITY"  # 匹配目标院校（大小写不敏感）


def _to_int(x):
    """将字符串或数值安全转换为整数(可空)。"""
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().replace(",", "")
    if s == "":
        return pd.NA
    try:
        return int(float(s))
    except Exception:
        return pd.NA


def _to_float(x):
    """将字符串或数值安全转换为浮点(可空)。"""
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().replace(",", "")
    if s == "":
        return pd.NA
    try:
        return float(s)
    except Exception:
        return pd.NA


def extract_one_file(xlsx_path: str) -> dict | None:
    """
    从单个学科文件中抽取目标院校所在行，返回统一字段的字典：
    {
        "Discipline", "Rank", "Institution", "Region",
        "Papers", "Citations", "CitesPerPaper", "HighlyCitedPapers"
    }
    若未找到则返回 None。
    """
    discipline = os.path.splitext(os.path.basename(xlsx_path))[0]

    # 读全部工作表，兼容不同 sheet 命名
    try:
        sheets = pd.read_excel(xlsx_path, sheet_name=None)
    except Exception as e:
        print(f"[WARN] 读取失败：{xlsx_path} -> {e}")
        return None

    for _, df in sheets.items():
        if df is None or df.empty:
            continue

        # 统一转为字符串作查找（不改变原 df 的数值类型）
        df_str = df.astype(str).applymap(lambda v: v.strip())

        # 优先做严格等值匹配（忽略大小写、首尾空白）
        eq_mask = df_str.apply(
            lambda col: col.str.fullmatch(TARGET, case=False, na=False),
            axis=0
        ).any(axis=1)

        # 如果没匹配上，再退化为包含匹配（容错）
        mask = eq_mask
        if not mask.any():
            mask = df_str.apply(
                lambda col: col.str.contains(re.escape(TARGET), case=False, na=False),
                axis=0
            ).any(axis=1)

        if not mask.any():
            continue

        # 取第一条命中行
        row = df.loc[mask].iloc[0]

        # 常见列位（ESI 下载常见是 7 列：Rank, Institution, Region, Papers, Citations, CitesPerPaper, HighlyCitedPapers）
        cols = list(df.columns)
        values = [row.get(c, pd.NA) for c in cols[:7]]

        rec = {
            "Discipline": discipline,
            "Rank": _to_int(values[0]),
            "Institution": str(values[1]).strip() if len(values) > 1 else "",
            "Region": str(values[2]).strip() if len(values) > 2 else "",
            "Papers": _to_int(values[3]) if len(values) > 3 else pd.NA,
            "Citations": _to_int(values[4]) if len(values) > 4 else pd.NA,
            "CitesPerPaper": _to_float(values[5]) if len(values) > 5 else pd.NA,
            "HighlyCitedPapers": _to_int(values[6]) if len(values) > 6 else pd.NA,
        }
        return rec

    return None


def main():
    print(f"[INFO] INPUT_DIR : {INPUT_DIR}")
    print(f"[INFO] OUTPUT_XLSX: {OUTPUT_XLSX}")

    # 1) 校验输入目录
    if not os.path.isdir(INPUT_DIR):
        print(f"[ERROR] 未找到目录：{INPUT_DIR}（请先解压 data.zip 到该目录）")
        sys.exit(1)

    # 2) 收集 .xlsx
    xlsx_files = [
        os.path.join(INPUT_DIR, fn)
        for fn in os.listdir(INPUT_DIR)
        if fn.lower().endswith(".xlsx")
    ]
    if not xlsx_files:
        print(f"[ERROR] 目录 {INPUT_DIR} 中未发现 .xlsx 文件")
        sys.exit(1)

    # 3) 逐文件抽取目标行
    records = []
    for fp in sorted(xlsx_files):
        rec = extract_one_file(fp)
        if rec is not None:
            records.append(rec)
        else:
            print(f"[INFO] 未在该文件中找到目标院校：{os.path.basename(fp)}")

    if not records:
        print("[ERROR] 所有文件均未找到目标院校，无法生成输出。")
        sys.exit(2)

    # 4) 汇总与导出
    df = pd.DataFrame(records).sort_values(by=["Discipline"]).reset_index(drop=True)

    # 确保输出目录存在
    out_dir = os.path.dirname(OUTPUT_XLSX) or "."
    os.makedirs(out_dir, exist_ok=True)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="analysis", index=False)

    print(f"[OK] 已生成：{OUTPUT_XLSX} ；共 {len(df)} 行。")


if __name__ == "__main__":
    main()
