import torch
from train.model import CharTransformer
from train.utils.tokenizer import chars, char2idx, idx2char

def load_model(path='train/models/webgen-scratch.pt'):
    model = CharTransformer(vocab_size=len(chars))
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def generate(model, prompt, max_len=500):
    idxs = [char2idx.get(c,0) for c in prompt]
    input_tensor = torch.tensor([idxs])
    output_str = prompt
    for _ in range(max_len):
        with torch.no_grad():
            logits = model(input_tensor)
            next_idx = torch.argmax(logits[0, -1]).item()
            next_char = idx2char[next_idx]
            output_str += next_char
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_idx]])], dim=1)
    return output_str
