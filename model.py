import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration,T5Tokenizer
import torch.nn.functional as F

class T5model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer=T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese")
        self.model=T5ForConditionalGeneration.from_pretrained("sonoisa/t5-base-japanese")
    
    def forward(self,input_ids,attention_mask=None,decoder_input_ids=None,
                decoder_attention_mask=None,labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        
    
    