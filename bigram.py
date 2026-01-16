import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# hyperparameters
batch_size = 32 # How many independent sequences we will process in parallel
block_size = 8 # The maximum "context" length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Find all the unique characters in our text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Map characters to integers (and back)
stringToInteger = { char:integer for integer, char in enumerate(chars) }
integerToString = { integer:char for integer, char in enumerate(chars) }

# Encode: take a string, output a list of integers
encode = lambda string: [stringToInteger[char] for char in string]

# Decode: take a list of integers, output a string
decode = lambda intList: ''.join([integerToString[integer] for integer in intList]) 

# Make training and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # Use first 90% for taining and use the last 10% for testing
train_data = data[:n]
eval_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of inputs x and targets y
    data = train_data if split == 'train' else eval_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Our simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) # (B, T, C)

        if targets is None:
            loss = None
        else:
            # Reshape logits so we can calculate loss
            B, T, C = logits.shape # batch, time, channels
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # loss function
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get prediction
            logits, loss = self(idx)
            # focus on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# Create our model
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Create a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # evaluate periodically
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, eval loss {losses['eval']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)

print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))