import torch
from torch.utils.data import Dataset
import pandas as pd


class PeptideDataset(Dataset):
    def __init__(self, csv_file, max_len=100):
        self.df = pd.read_csv(csv_file)
        self.df = self.df.dropna(subset=['Sequence'])
        self.max_len = max_len
        self.pad_idx = 0
        self.cls_idx = 1
        self.mask_idx = 2
        self.unk_idx = 3
        aas = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_dict = {a: i + 4 for i, a in enumerate(aas)}
        self.vocab_size = len(self.aa_dict) + 4

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = str(self.df.iloc[idx]['Sequence']).strip().upper()
        tokens = [self.cls_idx] + [self.aa_dict.get(aa, self.unk_idx) for aa in seq[:self.max_len]]
        pad_len = (self.max_len + 1) - len(tokens)
        if pad_len > 0:
            tokens.extend([self.pad_idx] * pad_len)

        return torch.tensor(tokens, dtype=torch.long)
