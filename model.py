import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
    """
    def __init__(self, d_model: int, dropout: float = 0., max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        if d_k is None:
            self.d_k = self.d_model // self.n_heads
        if d_v is None:
            self.d_v = self.d_model // self.n_heads
        self.W_Q = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(self.d_model, self.n_heads * self.d_v, bias=False)
        self.W_0 = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)

    def attention(self, Q, K, V):
        scale = math.sqrt(K.shape[1])
        return torch.softmax((Q @ K.transpose(-2, -1)) / scale, dim=-1) @ V

    def forward(self, X):
        batch_size = X.shape[0]
        # ----------------------------------------------------------------------
        # dimensions start from batch_size x n_heads
        Q = self.W_Q(X).view(batch_size, -1, self.n_heads, self.d_k).permute(2, 0, 1, 3)
        K = self.W_K(X).view(batch_size, -1, self.n_heads, self.d_k).permute(2, 0, 1, 3)
        V = self.W_V(X).view(batch_size, -1, self.n_heads, self.d_v).permute(2, 0, 1, 3)
        # ----------------------------------------------------------------------
        result = []
        for n_head in range(self.n_heads):
            result.append(self.attention(Q[n_head], K[n_head], V[n_head]).unsqueeze(0))  # batch_size x seq_len x d_v
        attention = torch.cat(result, dim=0)  # n_heads x batch_size x seq_len x d_v
        return self.W_0(attention.permute(1, 2, 0, 3).reshape(batch_size, -1, self.n_heads * self.d_v))


class FFTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm = nn.LayerNorm(normalized_shape=384)
        self.multi_head_attention = MultiHeadAttention(d_model=384, n_heads=2)
        self.conv1 = nn.Conv1d(in_channels=384, out_channels=1536, kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(in_channels=1536, out_channels=384, kernel_size=3, padding='same')
        self.relu = nn.ReLU()

    def forward(self, X):
        X_residual = X
        X = self.pre_norm(X)  # Pre LayerNorm
        X = self.multi_head_attention(X)
        X += X_residual
        X_residual = X
        X = self.pre_norm(X)  # Pre LayerNorm
        X = self.conv2(self.relu(self.conv1(X.transpose(-2, -1)))).transpose(-2, -1)
        X += X_residual
        return X


class DurationPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=384, out_channels=384, kernel_size=3, padding='same')
        self.activation1 = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(normalized_shape=384),
            nn.Dropout(0.)
        )
        self.conv2 = nn.Conv1d(in_channels=384, out_channels=384, kernel_size=3, padding='same')
        self.activation2 = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(normalized_shape=384),
            nn.Dropout(0.)
        )
        self.linear = nn.Linear(384, 1)

    def forward(self, X):
        X = self.activation1(self.conv1(X.transpose(-2, -1)).transpose(-2, -1))
        X = self.activation2(self.conv2(X.transpose(-2, -1)).transpose(-2, -1))
        return torch.exp(self.linear(X).squeeze(-1))


class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.duration_predictor = DurationPredictor()
        self.alpha = 1.0

    def forward(self, X, true_durations):
        batch_size = X.shape[0]
        durations = self.duration_predictor(X)
        if self.training:
            round_durations = torch.round(true_durations * self.alpha).int()
        else:
            round_durations = torch.round(durations * self.alpha).int()
        regularized = []
        for index in range(batch_size):
            regularized.append(torch.repeat_interleave(X[index], round_durations[index], dim=0))
        result = pad_sequence(regularized, batch_first=True)
        return result, durations


class FastSpeech(nn.Module):
    def __init__(self):
        super().__init__()
        self.phoneme_embedding = nn.Embedding(num_embeddings=51, embedding_dim=384)
        self.positional_encoding = PositionalEncoding(d_model=384)
        self.FFT1 = nn.Sequential(*[FFTBlock() for _ in range(6)])
        self.length_regulator = LengthRegulator()
        self.FFT2 = nn.Sequential(*[FFTBlock() for _ in range(6)])
        self.linear = nn.Linear(384, 80)

    def forward(self, X, durations):
        embedding = self.positional_encoding(self.phoneme_embedding(X))
        embedding = self.FFT1(embedding)
        fft1_reg, predicted_durations = self.length_regulator(embedding, durations)
        embedding = self.positional_encoding(fft1_reg)
        embedding = self.FFT2(embedding)
        return self.linear(embedding), predicted_durations
