import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, d_token):
        super().__init__()
        # Weight tensor of shape [num_features, d_token]
        self.weight = nn.Parameter(torch.empty(num_features, d_token))
        # Bias tensor of shape [num_features, d_token]
        self.bias = nn.Parameter(torch.empty(num_features, d_token))
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.uniform_(self.bias, -1.0 / (self.weight.size(1) ** 0.5), 1.0 / (self.weight.size(1) ** 0.5))

    def forward(self, x):
        # x shape: [batch_size, num_features]
        # x.unsqueeze(-1) shape: [batch_size, num_features, 1]
        # self.weight.unsqueeze(0) shape: [1, num_features, d_token]
        # Resulting token tensor shape: [batch_size, num_features, d_token]
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

class FTTransformer(nn.Module):
    def __init__(self, num_features, d_token=32, n_blocks=3, n_heads=4, d_ffn=64, dropout=0.1):
        super().__init__()
        assert d_token % n_heads == 0, "d_token must be divisible by n_heads"
        
        self.tokenizer = FeatureTokenizer(num_features, d_token)
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        
        # Classification head
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, 1)
        )
        
    def forward(self, x):
        # x shape: [batch_size, num_features]
        batch_size = x.size(0)
        
        # Tokenize features
        tokens = self.tokenizer(x)  # shape: [batch_size, num_features, d_token]
        
        # Expand CLS token to match batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # shape: [batch_size, 1, d_token]
        
        # Concatenate CLS token with feature tokens
        # shape: [batch_size, 1 + num_features, d_token]
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        # Pass through Transformer
        out_tokens = self.transformer(tokens)  # shape: [batch_size, 1 + num_features, d_token]
        
        # Extract the representation of the CLS token
        cls_out = out_tokens[:, 0, :]  # shape: [batch_size, d_token]
        
        # Apply LayerNorm and Classification Head
        cls_out = self.norm(cls_out)
        logits = self.head(cls_out)  # shape: [batch_size, 1]
        
        return logits.squeeze(-1)  # shape: [batch_size]
