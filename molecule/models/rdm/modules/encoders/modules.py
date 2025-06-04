import torch.nn as nn


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, c, key=None):
        c = self.embedding(c)
        return c

class ContinuousEmbedder(nn.Module):
    def __init__(self, embed_dim, input_dim=1, key="homo"):
        super().__init__()
        self.key = key
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, c, key=None):
        c = self.mlp(c)
        return c