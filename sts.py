import torch
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, LabelAccuracyEvaluator
import logging
from datetime import datetime
import sys
import re
import numpy as np
import argparse
from utils import read_sts

parser = argparse.ArgumentParser()
parser.add_argument('--model',dest='name_model', type=str, help='bert/roberta/gpt...', default='bert-base-cased')
parser.add_argument('--dataset', dest='dataset', type=str, help='name of dataset', default='')
parser.add_argument('--path-data',dest='path_data', type=str , help='path of folder data', default='')
parser.add_argument('--pooling',dest='pooling', type=str, help='mean/max', default='mean')
parser.add_argument('--use-topic',dest='use_topic', type=str, help='Use Trasformer-topic (True/False)', default='False')
parser.add_argument('--train-topic',dest='train_topic', type=str, help='Update topic embed during training (True/False)', default='False')
parser.add_argument('--train-file',dest='train_file', type=str, default='')
parser.add_argument('--dev-file', dest='dev_file', type=str, default='')
parser.add_argument('--test-file',dest='test_file', type=str, default='')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=8)
parser.add_argument('--epochs', dest='num_epochs', type=int, default=2)
parser.add_argument('--num-topics', dest='num_topic', type=int, default=0)
args = parser.parse_args()



use_topic = True if args.use_topic.lower() != 'false' else False
train_topic = True if args.train_topic.lower() != 'false' else False

# model_save_path = args.path_data+'/'+args.dataset+'_'+args.name_model+'_'+str(args.use_topic)+'_'+str(args.train_topic)+'_'+args.pooling +"_"+ str(args.num_topic)

# topic set up
if use_topic is True:
    model_save_path = args.path_data+'/'+args.dataset+'_'+args.name_model+'_'+str(args.use_topic)+'_'+str(args.train_topic)+'_'+args.pooling +"_"+ str(args.num_topic)
    W = torch.tensor(np.load(args.path_data+'/W_gensim_t'+ str(args.num_topic)  + '.npy'), dtype = torch.float) 
    word_embedding_model = models.Transformer_Topic(args.name_model, topic_weight = W, train_topic = train_topic, max_seq_length = 512)
    transfer_layer = models.Features_transfer(word_embedding_model.get_word_embedding_dimension(), word_embedding_model.get_word_embedding_dimension())
    pooling_model = models.Pooling(transfer_layer.get_word_embedding_dimension(), pooling_mode_mean_tokens=True, pooling_mode_cls_token=False, pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[word_embedding_model, transfer_layer, pooling_model])
else:
    model_save_path = args.path_data+'/'+args.dataset+'_'+args.name_model+'_'+str(args.use_topic)+'_'+str(args.train_topic)+'_'+args.pooling
    word_embedding_model = models.Transformer(args.name_model, max_seq_length = 256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True, pooling_mode_cls_token=False, pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model.cuda()

train_samples = []
train_data = read_sts(args.path_data+'/'+args.train_file)
for sample in train_data:
    train_samples.append(InputExample(texts=[sample[1], sample[2]], label=float(sample[0])/5.0))
        
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

dev_samples = []
if args.dev_file == '':
    dev_samples = train_samples[0:1000]
else:
    train_data = read_sts(args.path_data+'/'+args.dev_file)
    for sample in train_data:
        dev_samples.append(InputExample(texts=[sample[1], sample[2]], label=float(sample[0])/5.0))

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=args.num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )


test_samples = []
train_data = read_sts(args.path_data+'/'+args.test_file)
for sample in train_data:
    test_samples.append(InputExample(texts=[sample[1], sample[2]], label=float(sample[0])/5.0))
    
model = SentenceTransformer(model_save_path)
model.cuda()
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=args.batch_size, name='sts-test')
test_evaluator(model, output_path=model_save_path)
test_evaluator.predict(model, output_path=model_save_path)