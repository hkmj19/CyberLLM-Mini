import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Load & Preprocess Data
def load_data(file_path: str = "data.txt"):
    """Load text data and build vocabulary."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    words = text.split()
    vocab = sorted(set(words))
    if "<unk>" not in vocab:
        vocab.append("<unk>")
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    data = [word2idx.get(w, word2idx["<unk>"]) for w in words]
    return data, word2idx, idx2word, vocab


data, word2idx, idx2word, vocab = load_data("data.txt")
vocab_size = len(vocab)
seq_len = 5  # Context window size

# Dataset Class
class WordDataset(Dataset):
    """Dataset for word-level next-token prediction."""
    def __init__(self, data, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return torch.tensor(x), torch.tensor(y)


dataset = WordDataset(data, seq_len)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Tiny Transformer Model
class TinyWordTransformer(nn.Module):
    """Small Transformer for word-level language modeling."""
    def __init__(self, vocab_size: int, n_embd: int = 64, n_heads: int = 2, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_heads,
            dim_feedforward=128,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(n_embd, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x) * (self.embed.embedding_dim ** 0.5)
        x = x.permute(1, 0, 2)  # (seq_len, batch, emb)
        out = self.transformer(x)
        out = out.permute(1, 0, 2)  # (batch, seq_len, emb)
        logits = self.fc(out)
        return logits


model = TinyWordTransformer(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training
epochs = 200
print("Training model...")
for epoch in range(epochs):
    total_loss = 0.0
    for x, y in dataloader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}")

print("Training completed!")

# Text Generation Function
def generate(model, start_words: str, max_words: int = 30, temperature: float = 0.8, top_k: int = 5) -> str:
    """Generate text continuation from starting words."""
    model.eval()
    words_generated = start_words.lower().split()
    words_generated = [w if w in word2idx else "<unk>" for w in words_generated]

    for _ in range(max_words):
        context = words_generated[-seq_len:]
        input_idx = torch.tensor([[word2idx.get(w, word2idx["<unk>"]) for w in context]])
        logits = model(input_idx)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        top_probs, top_idx = torch.topk(probs, top_k)
        next_idx = top_idx[0, torch.multinomial(top_probs[0], 1)]
        next_word = idx2word[next_idx.item()]
        words_generated.append(next_word)

    return " ".join(words_generated)


# Interactive User Input
print("\nType 'quit' to exit.\n")
while True:
    start = input("Enter starting words: ")
    if start.lower() == "quit":
        break
    output = generate(model, start, max_words=30)
    print("\nGenerated text:\n", output)
