import torch
from transformers import AutoTokenizer
        
def tokenize(df, tokenizer_name: str, token_max_length: int = None, random_state: int = 42):
    """_summary_

    Returns:
        _type_: _description_
    """
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if not token_max_length:
        return df.map(lambda x: tokenizer(x["text"], return_tensors="pt"))  #  For EDA purposes
    else:
        return df.map(lambda x: tokenizer(x["text"], return_tensors="pt", truncation=True, padding=True, max_length=token_max_length), batched=True)



