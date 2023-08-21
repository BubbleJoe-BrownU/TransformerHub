"""
Load the Shakespeare dataset for character-level language modeling.


"""

import os
import pickle
import requests
import numpy as np

def main():
    dataset_path = os.path.join(os.path.dirname(__file__), 'shakespeare.txt')
    if not os.path.exists(dataset_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(dataset_path, "w") as f:
            f.write(requests.get(data_url).text)
    
    with open(dataset_path, "r") as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")
    
    # get the set of unique characters that occur in the dataset, i.e. the vocabulary
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("vocabulary:", ''.join(chars))
    print("vocab size: ", vocab_size)
    
    # create a mapping from characters to integers
    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for i, ch in enumerate(chars)}
    
    def encode(s):
        return [stoi[c] for c in s]
    
    def decode(l):
        return ''.join([itos[i] for i in l])
    
    train_data = data[:int(len(data)*0.9)]
    val_data = data[int(len(data)*0.9):]
    
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train split has {len(train_ids):,} tokens")
    print(f"val split has {len(val_ids):,} tokens")
    
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
    
    meta = dict(
        vocab_size=vocab_size,
        itos=itos,
        stoi=stoi,
    )
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens

if __name__ == "__main__":
    main()