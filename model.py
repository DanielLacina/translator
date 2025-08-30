import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, sequence_length, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(sequence_length, 1, embedding_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, embedding): 
        out1 = embedding + self.pe
        return self.dropout(out1)

class MultiAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()  
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.embedding_dim = embedding_dim  
        self.Wk = nn.Linear(embedding_dim, embedding_dim)
        self.Wq = nn.Linear(embedding_dim, embedding_dim)
        self.Wv = nn.Linear(embedding_dim, embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, embedding):
        batch_size = embedding.shape[0]
        K = self.Wk(embedding) 
        Q = self.Wq(embedding)
        V = self.Wv(embedding)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        similarities = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(self.head_dim)
        probabilities = torch.softmax(similarities, dim=-1)
        output = torch.matmul(probabilities, V)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.embedding_dim)
        output = self.fc_out(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.layer1 = nn.Linear(embedding_dim, embedding_dim*4)
        self.layer2 = nn.Linear(embedding_dim*4, embedding_dim)

    def forward(self, embedding):
        output = self.layer1(embedding)
        output = torch.relu(output)
        output = self.layer2(output)
        return output


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sequence_length, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, sequence_length)
        self.multi_attention = MultiAttention(embedding_dim, num_heads)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.feed_forward = FeedForward(embedding_dim)

    def forward(self, sequence):
        embedding = self.embedding(sequence)
        embedding_with_pos = self.pos_encoder(embedding)
        new_embeddings = self.multi_attention(embedding_with_pos)
        output = new_embeddings + embedding_with_pos
        layer_norm = self.layer_norm(output)
        feed_forward = self.feed_forward(layer_norm)  
        output = feed_forward + layer_norm
        layer_norm = self.layer_norm(output)
        return layer_norm
   
