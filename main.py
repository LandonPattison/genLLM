from multiprocessing import freeze_support
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
import random
import sys
import os
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F
from typing import Tuple
from torch.nn.utils.rnn import pad_sequence
import json

device = "cuda"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:8000'

# Tokenize the dataset
def tokenize_data(data: str):
    tokenizer = get_tokenizer("basic_english")
    return tokenizer(data)

# Split the dataset into training and validation sets
def split_data(data: str, split_ratio: float = 0.8) -> Tuple[str, str]:
    lines = data.strip().split('\n')
    random.shuffle(lines)
    num_train = int(len(lines) * split_ratio)
    return '\n'.join(lines[:num_train]), '\n'.join(lines[num_train:])

# Create the vocabulary
def create_vocab(train_data: str):
    vocab = build_vocab_from_iterator(map(tokenize_data, train_data.split('\n')), specials=['<unk>', '<pad>', '<sos>', '<eos>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab

# Custom dataset class
class TextCompletionDataset(Dataset):
    def __init__(self, data: str, vocab):
        self.data = data
        self.vocab = vocab
        self.tokenizer = get_tokenizer("basic_english")

    def __len__(self):
        return len(self.data.split('\n'))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        line = self.data.split('\n')[idx]
        tokens = ['<sos>'] + self.tokenizer(line) + ['<eos>']
        input_ids = torch.tensor([self.vocab[token] for token in tokens], dtype=torch.long)
        return input_ids[:-1], input_ids[1:]

def collate_fn(batch, vocab):
    src = [x[0] for x in batch]
    tgt = [x[1] for x in batch]
    src = pad_sequence(src, batch_first=True, padding_value=vocab['<pad>'])
    tgt = pad_sequence(tgt, batch_first=True, padding_value=vocab['<pad>'])
    return src, tgt

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiheadSelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, query, key, value, attn_mask=None):
        return self.multihead_attn(query, key, value, attn_mask=attn_mask)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadSelfAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src = src + self.dropout(self.self_attn(src, src, src, attn_mask=src_mask)[0])
        src = self.norm1(src)
        src = src + self.dropout(self.feed_forward(src))
        src = self.norm2(src)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadSelfAttention(d_model, nhead, dropout)
        self.src_attn = MultiheadSelfAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt = tgt + self.dropout(self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0])
        tgt = self.norm1(tgt)
        tgt = tgt + self.dropout(self.src_attn(tgt, memory, memory, attn_mask=memory_mask)[0])
        tgt = self.norm2(tgt)
        tgt = tgt + self.dropout(self.feed_forward(tgt))
        tgt = self.norm3(tgt)
        return tgt

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, d_ff, dropout=0.1, max_len=5000):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.transformer_encoder = TransformerEncoderLayer(d_model, nhead, d_ff, dropout)
        self.transformer_decoder = TransformerDecoderLayer(d_model, nhead, d_ff, dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.num_layers = num_layers
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)

        for _ in range(self.num_layers):
            src = self.transformer_encoder(src, src_mask)

        memory = src
        for _ in range(self.num_layers):
            tgt = self.transformer_decoder(tgt, memory, tgt_mask, memory_mask)

        return self.fc_out(tgt)

def train_loop(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)

        # Forward pass
        output = model(src, tgt[:-1])
        output = output.view(-1, output.shape[-1])

        # Calculate loss
        loss = criterion(output, tgt[1:].view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update weights
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Free up unused GPU memory
        torch.cuda.empty_cache()

    return total_loss / len(train_loader)



# Evaluation loop
def eval_loop(model, val_loader, criterion, device):
    print("in eval loop")
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)

            # Forward pass
            output = model(src, tgt[:-1])
            output = output.view(-1, output.shape[-1])

            # Calculate loss
            loss = criterion(output, tgt[1:].view(-1))

            total_loss += loss.item()

    return total_loss / len(val_loader)

def tokenize(text):
    return [token for token in text.split(' ') if token]

def detokenize(tokens):
    return ' '.join(tokens)

# Function for generating text completions
def generate_completion(model, vocab, input_text, max_length=1000, top_k=5, temperature=0.8):
    input_tokens = tokenize(input_text)
    input_tensor = torch.tensor([vocab[token] for token in input_tokens], dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_length):
        with torch.no_grad():
            output = model(input_tensor, input_tensor)
            output = output.squeeze(0)

        logits = output[-1] / temperature
        top_k_indices = torch.topk(logits, k=top_k).indices

        # Randomly select one of the top k tokens
        next_token = top_k_indices[torch.randint(0, top_k, (1,)).item()].item()
        input_tokens.append(vocab.lookup_token(next_token))
        input_tensor = torch.tensor([vocab[token] for token in input_tokens], dtype=torch.long).unsqueeze(0).to(device)

        # # Check if the generated token is an end-of-sentence token
        # if next_token == vocab['<eos>']:
        #     break

    generated_text = detokenize(input_tokens)
    return generated_text


def main(): 
    torch.cuda.empty_cache()

    dataset_path = "data.txt"
    split_ratio = 0.8

    print("Uploading data")
    # Load the dataset and preprocess it
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = f.read()

    train_data_raw, val_data_raw = split_data(raw_data, split_ratio)
    vocab = create_vocab(train_data_raw)

    torch.save(vocab, 'vocab.pt')

    # Create data loaders
    train_dataset = TextCompletionDataset(train_data_raw, vocab)
    val_dataset = TextCompletionDataset(val_data_raw, vocab)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=lambda batch: collate_fn(batch, vocab))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=lambda batch: collate_fn(batch, vocab))
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    vocab_size = len(vocab)
    d_model = 512
    nhead = 8
    num_layers = 6
    d_ff = 2048
    dropout = 0.1
    max_len = 5000

    print("we go model")
    model = TransformerModel(vocab_size, d_model, nhead, num_layers, d_ff, dropout, max_len).to(device)

    # Set the learning rate scheduler
    learning_rate = 0.0005
    warmup_steps = 4000

    def lr_lambda(step: int):
        return d_model ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * warmup_steps ** (-1.5))

    scheduler = optim.lr_scheduler.LambdaLR(optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9), lr_lambda)

    # Create the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

    import time

    # Train and evaluate the model
    num_epochs = 1
    best_val_loss = float('inf')
    best_model = None
    save_model_path = 'best_transformer_model.pth'

    print("startinge epoch")
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss = train_loop(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss = eval_loop(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            torch.save(best_model, save_model_path)

        elapsed_time = time.time() - start_time

        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed_time:.2f}s')

    print("Training completed.")

    # Load the trained model
    loaded_model = TransformerModel(vocab_size, d_model, nhead, num_layers, d_ff, dropout, max_len).to(device)
    loaded_model.load_state_dict(torch.load('best_transformer_model.pth'))
    loaded_model.eval()
    
if __name__ == "__main__":
    freeze_support()
    main()