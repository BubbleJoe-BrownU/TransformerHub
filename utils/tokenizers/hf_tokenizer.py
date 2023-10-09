from transformers import AutoTokenizer

def load_tokenizer(checkpoint):
    return AutoTokenizer.from_pretrained(checkpoint)