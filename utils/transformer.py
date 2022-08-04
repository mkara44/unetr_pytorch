import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim // head_num) ** (1 / 2)

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)

        query, key, value = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_attention(x)

        return x


class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x
        x = self.layer_norm1(x)

        _x = self.mlp(x)
        x = x + _x
        x = self.layer_norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num, z_idx_list):
        super().__init__()

        self.z_idx_list = z_idx_list

        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x):
        z_outputs = []
        for idx, layer_block in enumerate(self.layer_blocks, start=1):
            x = layer_block(x)
            if idx in self.z_idx_list:
                z_outputs.append(x)

        return z_outputs


class AbsPositionalEncoding1D(nn.Module):
    def __init__(self, tokens, dim):
        super().__init__()
        self.abs_pos_enc = nn.Parameter(torch.randn(1, tokens, dim))

    def forward(self, x):
        batch = x.size()[0]

        return x + repeat(self.abs_pos_enc, 'b ... -> (b tile) ...', tile=batch // self.abs_pos_enc.shape[0])


class Transformer3D(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_size, z_idx_list):
        super().__init__()

        self.patch_size = patch_size
        self.n_patches = int((img_dim[0] * img_dim[1] * img_dim[2]) / (patch_size ** 3))

        self.patch_embeddings = nn.Conv3d(in_channels, embedding_dim,
                                          kernel_size=patch_size, stride=patch_size, bias=False)

        self.position_embeddings = AbsPositionalEncoding1D(self.n_patches, embedding_dim)
        self.dropout = nn.Dropout(0.1)

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num, z_idx_list)

    def forward(self, x):
        embeddings = rearrange(self.patch_embeddings(x), 'b d x y z -> b (x y z) d')
        embeddings = self.position_embeddings(embeddings)
        embeddings = self.dropout(embeddings)

        z_outputs = self.transformer(embeddings)

        return z_outputs


if __name__ == '__main__':
    trans = Transformer3D(img_dim=(128, 128, 128),
                          in_channels=4,
                          patch_size=16,
                          embedding_dim=768,
                          block_num=12,
                          head_num=12,
                          mlp_dim=3072,
                          z_idx_list=[3, 6, 9, 12])
    z3, z6, z9, z12 = trans(torch.rand(1, 4, 128, 128, 128))
    print(z3.shape)
    print(z6.shape)
    print(z9.shape)
    print(z12.shape)
