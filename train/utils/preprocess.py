import json
from tokenizer import char2idx

def load_dataset(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            prompt = [char2idx.get(c,0) for c in obj['prompt']]
            output = [char2idx.get(c,0) for c in obj['output']]
            data.append((prompt, output))
    return data
