import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from main import TransformerModel
from main import generate_completion
import json
from torchtext.vocab import Vocab

vocab = torch.load('vocab.pt')

device = "cuda"

# Instantiate the model
vocab_size = len(vocab)
d_model = 512
nhead = 8
num_layers = 6
d_ff = 2048
dropout = 0.1
max_len = 5000

# Load the trained model
loaded_model = TransformerModel(vocab_size, d_model, nhead, num_layers, d_ff, dropout, max_len).to(device)
loaded_model.load_state_dict(torch.load('best_transformer_model.pth'))
loaded_model.eval()

def tokenize(text):
    return [token for token in text.split(' ') if token]

def detokenize(tokens):
    return ' '.join(tokens)

# Test the text completion function
input_text = "sebastion: "
completion = generate_completion(loaded_model, vocab, input_text)
print(f"Generated completion: {completion}")
