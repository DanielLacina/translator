import pandas as pd
from torchtext.data import get_tokenizer
import torch
import json

SEQUENCE_LENGTH = 512
if __name__ == "__main__":
    tokenizer = get_tokenizer("basic_english")
    word_indices = {}
    word_indices["<pad>"] = 0
    word_indices["<unk>"] = 1

    sequences = []
    with open("data/europarl-v7.es-en.en", "r", encoding="utf-8") as en_f:
        sequence = []
        for line in en_f:
            words = tokenizer(line.strip())
            for word in words:
                if word not in word_indices:  
                    word_indices[word] = len(word_indices)
                sequence.append(word_indices[word])
            while len(sequence) < SEQUENCE_LENGTH:
                sequence.append(0)
            sequences.append(sequence[:SEQUENCE_LENGTH])


    sequences = torch.tensor(sequences, dtype=torch.long)
    torch.save({"sequences": sequences}, "sequences.pt")

    with open("word_indices.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(word_indices, ensure_ascii=False, indent=4))

    with open("word_count.txt", "w", encoding="utf-8") as f:
        f.write(str(len(word_indices)))

