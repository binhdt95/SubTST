from torch import nn
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
import json
from typing import List, Dict, Optional, Union, Tuple
import os
import numpy as np
import re

class Transformer_Topic(nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.

    :param model_name_or_path: Huggingface models name (https://huggingface.co/models)
    :param topic_path: Topic models path 
    :param max_seq_length: Truncate any inputs longer than max_seq_length
    :param model_args: Arguments (key, value pairs) passed to the Huggingface Transformers model
    :param cache_dir: Cache dir for Huggingface Transformers to store/load models
    :param tokenizer_args: Arguments (key, value pairs) passed to the Huggingface Tokenizer model
    :param do_lower_case: Lowercase the input
    """
    def __init__(self, model_name_or_path: str, topic_weight, train_topic: bool = False, max_seq_length: int = 128,
                 model_args: Dict = {}, cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, do_lower_case: Optional[bool] = None):
        super(Transformer_Topic, self).__init__()
        self.config_keys = ['max_seq_length']
        self.max_seq_length = max_seq_length
        self.train_topic = train_topic
        if do_lower_case is not None:
            tokenizer_args['do_lower_case'] = do_lower_case
        
        self.W = topic_weight
        num_topic_embeddings, topic_embeddings_dimension = self.W.size()
        self.topic_embedding = nn.Embedding(num_topic_embeddings, topic_embeddings_dimension)
        self.topic_embedding.load_state_dict({'weight': self.W})
        self.topic_embedding.weight.requires_grad = self.train_topic
        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir, **tokenizer_args)
        
          
    def load_token_topic_embed(self, transfomer_ids):
        token_topic_embed = self.topic_embedding(transfomer_ids)
        return token_topic_embed
    
    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]
        token_ids = features['input_ids']
        token_topic_embeds = self.load_token_topic_embed(token_ids)
        output_tokens = torch.cat((output_tokens, token_topic_embeds),2)
        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens, 'attention_mask': features['attention_mask']})
        
        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})
        return features

    def get_word_embedding_dimension(self) -> int:
        return (self.auto_model.config.hidden_size) + (self.W.shape[1])

    def tokenize(self, texts: Union[List[str], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output['text_keys'] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output['text_keys'].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        output.update(self.tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length))
        return output


    def get_config_dict(self):
        config_dict = {key: self.__dict__[key] for key in self.config_keys}
        config_dict['train_topic'] = self.train_topic
        return config_dict

    def save(self, output_path: str):
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        # torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))
        torch.save(self.topic_embedding.weight, os.path.join(output_path, 'topic_embed.bin'))
        print(self.get_config_dict())
        with open(os.path.join(output_path, 'other.txt'), 'w') as fout:
            fout.write('dim: '+ str(self.get_word_embedding_dimension())+ '\n')
            fout.write('num_topics: '+ str(self.W.shape[1])+ '\n')
        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break
        # weights = torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu'))
        # topic_weight = weights['topic_embedding.weight']
        topic_weight = torch.load(os.path.join(input_path, 'topic_embed.bin'), map_location=torch.device('cpu'))
        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return Transformer_Topic(model_name_or_path=input_path, topic_weight = topic_weight, **config)






