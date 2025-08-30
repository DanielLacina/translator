import torch
from model import Encoder
from torch.utils.data import DataLoader, TensorDataset
from preprocessing import SEQUENCE_LENGTH

device = torch.device("cuda")
data = torch.load("sequences.pt")
sequences = data["sequences"].to(device)
with open("word_count.txt", "r", encoding="utf-8") as f:
    vocab_size = int(f.read().strip())
embedding_dim = 512
num_heads = 8
model = Encoder(vocab_size, embedding_dim, SEQUENCE_LENGTH, num_heads).to(device)
dataset = TensorDataset(sequences)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

for [sequences] in dataloader: 
    print(model(sequences))


