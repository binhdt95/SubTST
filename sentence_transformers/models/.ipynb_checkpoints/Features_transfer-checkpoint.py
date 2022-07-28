import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import os
import json
from ..util import fullname, import_from_string

class Features_transfer(nn.Module):
    
    
    def __init__(self, in_features: int, out_features: int, activation_function=nn.Tanh(), hidden_dropout_prob: int = 0.1):
        super(Features_transfer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dropout_prob = hidden_dropout_prob
        self.activation_function = activation_function
        
        self.dense = nn.Linear(self.in_features, self.out_features)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.norm = nn.LayerNorm(self.out_features, eps = 1e-12)

    
    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        
        token_embeddings = self.dense(token_embeddings)
        token_embeddings = self.dropout(token_embeddings)
        token_embeddings = self.norm(token_embeddings)
        
        cls_tokens = token_embeddings[:, 0, :]  # CLS token is first token
        features.update({'token_embeddings': token_embeddings, 'cls_token_embeddings': cls_tokens})
        
        return features
        
        
    def get_word_embedding_dimension(self) -> int:
        return self.out_features

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump({'in_features': self.in_features, 'out_features': self.out_features, 'activation_function': fullname(self.activation_function)}, fOut)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        config['activation_function'] = import_from_string(config['activation_function'])()
        model = Features_transfer(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))
        return model
