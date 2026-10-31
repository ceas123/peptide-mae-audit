import numpy as np
import torch
import torch.nn as nn
from einops import repeat
from timm.models.vision_transformer import Block
from timm.layers import trunc_normal_


def random_indexes(size, generator=None):
    forward_indexes = np.arange(size)
    if generator is None:
        np.random.shuffle(forward_indexes)
    else:
        generator.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 1, repeat(indexes, 'b t -> b t c', c=sequences.shape[-1]))


class PatchShuffle1D(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, patches, generator=None):
        B, T, C = patches.shape
        remain_T = int(T * (1.0 - self.ratio))
        if remain_T >= T:
            fwd = torch.arange(T, device=patches.device).unsqueeze(0).expand(B, T).contiguous()
            return patches, fwd, fwd.clone(), T
        indexes = [random_indexes(T, generator) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=0),
                                          dtype=torch.long, device=patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=0),
                                           dtype=torch.long, device=patches.device)
        patches = take_indexes(patches, forward_indexes)
        patches = patches[:, :remain_T, :]
        return patches, forward_indexes, backward_indexes, remain_T


class MAE_Encoder1D(nn.Module):
    def __init__(self, vocab_size=24, max_len=100, dim=256, num_layer=6,
                 num_head=8, mask_ratio=0.75):
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len + 1, dim))
        self.shuffle = PatchShuffle1D(mask_ratio)
        self.transformer = nn.Sequential(*[Block(dim, num_head, qkv_bias=True)
                                           for _ in range(num_layer)])
        self.layer_norm = nn.LayerNorm(dim)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, tokens, generator=None):
        x = self.token_emb(tokens)
        x = x + self.pos_embedding
        cls_token = x[:, 0:1, :]
        seq_tokens = x[:, 1:, :]
        seq_tokens, _, backward_indexes, remain_T = self.shuffle(seq_tokens, generator)
        x = torch.cat([cls_token, seq_tokens], dim=1)
        x = self.layer_norm(self.transformer(x))
        return x, backward_indexes, remain_T

    def encode_full(self, tokens):
        x = self.token_emb(tokens)
        x = x + self.pos_embedding
        return self.layer_norm(self.transformer(x))


class MAE_Decoder1D(nn.Module):
    def __init__(self, vocab_size=24, max_len=100, dim=256, num_layer=4, num_head=8):
        super().__init__()
        self.max_len = max_len
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len + 1, dim))
        self.transformer = nn.Sequential(*[Block(dim, num_head, qkv_bias=True)
                                           for _ in range(num_layer)])
        self.head = nn.Linear(dim, vocab_size)
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        B = features.shape[0]
        T_remain = features.shape[1] - 1
        cls_token = features[:, 0:1, :]
        seq_features = features[:, 1:, :]
        mask_tokens = self.mask_token.expand(B, self.max_len - T_remain, -1)
        full_seq = torch.cat([seq_features, mask_tokens], dim=1)
        full_seq = take_indexes(full_seq, backward_indexes)
        x = torch.cat([cls_token, full_seq], dim=1)
        x = x + self.pos_embedding
        x = self.transformer(x)
        logits = self.head(x[:, 1:, :])
        return logits


class MAE_Peptide(nn.Module):
    def __init__(self, vocab_size=24, max_len=100, dim=256,
                 encoder_layer=6, encoder_head=8,
                 decoder_layer=4, decoder_head=8, mask_ratio=0.75):
        super().__init__()
        assert dim % encoder_head == 0
        assert dim % decoder_head == 0
        self.max_len = max_len
        self.encoder = MAE_Encoder1D(vocab_size, max_len, dim, num_layer=encoder_layer,
                                     num_head=encoder_head, mask_ratio=mask_ratio)
        self.decoder = MAE_Decoder1D(vocab_size, max_len, dim, num_layer=decoder_layer,
                                     num_head=decoder_head)

    def forward(self, tokens):
        features, backward_indexes, remain_T = self.encoder(tokens)
        logits = self.decoder(features, backward_indexes)
        B = tokens.shape[0]
        shuffled_mask = torch.ones(B, self.max_len, dtype=torch.bool, device=tokens.device)
        shuffled_mask[:, :remain_T] = False
        loss_mask = torch.gather(shuffled_mask, 1, backward_indexes)
        return logits, loss_mask
