import torch
from torch.utils.data import DataLoader, Dataset
from model import CharTransformer
from utils.preprocess import load_dataset
from utils.tokenizer import chars, char2idx, idx2char

class CodeDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        prompt, output = self.data[idx]
        return torch.tensor(prompt), torch.tensor(output)

data = load_dataset('data/dataset.jsonl')
dataset = CodeDataset(data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = CharTransformer(vocab_size=len(chars))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(5):
    for prompts, outputs in dataloader:
        optimizer.zero_grad()
        preds = model(prompts)
        loss = criterion(preds.view(-1, len(chars)), outputs.view(-1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete")

torch.save(model.state_dict(), 'models/webgen-scratch.pt')
