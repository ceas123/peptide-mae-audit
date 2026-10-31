import os
import re
import pandas as pd


SOURCE_FILE = "222.xlsx"
OUTPUT_FILE = "snedataset.csv"
SEQ_COL = "Sequence"
ACT_COL = "Activity"
MIN_LEN, MAX_LEN = 5, 100


def find_col(df, target):
    for c in df.columns:
        if str(c).strip().lower() == target.lower():
            return c
    return None


def assign_label(text):
    s = str(text).lower()
    has_fungi = 'antifungal' in s
    has_bact = 'antibacterial' in s

    if has_fungi and not has_bact:
        return 'Antifungal'
    if has_bact and not has_fungi:
        return 'Antibacterial'
    return 'Other'



def build():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(base_dir, SOURCE_FILE)
    out_path = os.path.join(base_dir, OUTPUT_FILE)

    print("================ 构建 t-SNE 评估集 ================")
    df = pd.read_excel(src_path, dtype=str)

    seq_col = find_col(df, SEQ_COL)
    act_col = find_col(df, ACT_COL)
    if seq_col is None or act_col is None:
        raise ValueError(f"未找到列，实际列名: {list(df.columns)}")

    df = df[[seq_col, act_col]].copy()
    df.columns = ['Sequence', 'ActivityRaw']
    df = df.dropna(subset=['Sequence'])
    print(f"1. 原始读取: {len(df)} 条")

    df['Sequence'] = df['Sequence'].astype(str).str.strip().str.upper()
    df = df[df['Sequence'] != '']

    invalid_pattern = re.compile(r'[^ACDEFGHIKLMNPQRSTVWY]')
    valid_mask = ~df['Sequence'].apply(lambda x: bool(invalid_pattern.search(x)))
    df = df[valid_mask]
    print(f"2. 剔除非标准氨基酸后: {len(df)} 条")

    lengths = df['Sequence'].apply(len)
    df = df[(lengths >= MIN_LEN) & (lengths <= MAX_LEN)]
    print(f"3. 长度过滤({MIN_LEN}-{MAX_LEN})后: {len(df)} 条")

    df = df.drop_duplicates(subset=['Sequence'], keep='first').reset_index(drop=True)
    print(f"4. 去重后: {len(df)} 条")

    df['Label'] = df['ActivityRaw'].apply(assign_label)
    df = df[df['Label'] != 'Other'].reset_index(drop=True)
    print(f"5. 剔除无法归类(Other)后: {len(df)} 条")

    print("\n类别分布:")
    print(df['Label'].value_counts())

    result = df[['Sequence', 'Label']]
    result.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n完成，已写入: {out_path}")


if __name__ == "__main__":
    build()
