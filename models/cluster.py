import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FFN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Cluster(nn.Module):
    def __init__(self, num_token=4, token_dim=384, num_layers=8, num_heads=8):
        super(Cluster, self).__init__()
        self.num_token = num_token
        self.token_dim = token_dim

        self.tokens = nn.Embedding(num_token, token_dim)
        # trunc_normal_(self.tokens.weight, std=0.02)

        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(token_dim, num_heads) for _ in range(num_layers)
        ])
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(token_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(token_dim) for _ in range(num_layers)])
        self.ffn_layers = nn.ModuleList([FFN(token_dim, token_dim * 4) for _ in range(num_layers)])

    def forward(self, x):
        b, n, c = x.shape
        query = self.tokens.weight.unsqueeze(0).expand(b, -1, -1) # b, num_token, c
        query_seq_first = query.permute(1, 0, 2) # num_token, b, c
        x_seq_first = x.permute(1, 0, 2) # n, b, c

        for cross_attn, self_attn, norm, ffn in zip(self.cross_attn_layers, self.self_attn_layers, self.norm_layers, self.ffn_layers):
            query = query + cross_attn(query_seq_first, x_seq_first, x_seq_first)[0].permute(1, 0, 2)
            query = query + self_attn(query_seq_first, query_seq_first, query_seq_first)[0].permute(1, 0, 2)
            query = query + ffn(query)    
            query = norm(query)        

        query = F.normalize(query, p=2, dim=-1)
        x = F.normalize(x, p=2, dim=-1)
        similarity = torch.matmul(x, query.transpose(1, 2)) # b, n, num_token
        similarity = similarity + 1

        return similarity

