"""
Functions used in miniViT module
Reference Paper: AdaBins: Depth Estimation using Adaptive Bins (Shariq Farooq Bhat, Ibraheem Alhashim, Peter Wonka) https://arxiv.org/abs/2011.14141
Reference Code: https://github.com/shariqfarooq123/AdaBins
"""
import torch
import torch.nn as nn


class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=10, embedding_dim=128, num_heads=4):
        super(PatchTransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)  # takes shape S,N,E

        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)

        # self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim), requires_grad=True)
        self.positional_encodings = False
        self.embedding_dim = embedding_dim

    def forward(self, x):
        embeddings = self.embedding_convPxP(x).flatten(2)  # .shape = n,c,s = n, embedding_dim, s
        
        if not self.positional_encodings:
            self.positional_encodings = nn.Parameter(torch.rand(embeddings.shape[2], self.embedding_dim), requires_grad=True)

        # embeddings = nn.functional.pad(embeddings, (1,0))  # extra special token at start ?
        # print(embeddings.shape)
        # print(self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0).shape)
        embeddings = embeddings + self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0)

        # change to S,N,E format required by transformer
        embeddings = embeddings.permute(2, 0, 1)
        x = self.transformer_encoder(embeddings)  # .shape = S, N, E
        return x


class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, K):
        n, c, h, w = x.size()
        _, cout, ck = K.size()
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))  # .shape = n, hw, cout
        return y.permute(0, 2, 1).view(n, cout, h, w)
