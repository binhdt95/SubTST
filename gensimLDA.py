import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import models, losses
import numpy as np
import json

# data = []
# with open('run/data/vnPara/data.tsv', 'r') as fin:
#     for line in fin:
#         parts = line.strip().split('\t')
#         data.append(parts[0])
#         data.append(parts[1])

# with open('data/SICK/dev.csv', 'r') as fin:
#     for line in fin:
#         parts = line.strip().split('\t')
#         data.append(parts[1])
#         data.append(parts[2])

# with open('data/SICK/test.csv', 'r') as fin:
#     for line in fin:
#         parts = line.strip().split('\t')
#         data.append(parts[1])
#         data.append(parts[2])


# word_embedding_model = models.Transformer("bert-base-multilingual-cased" , max_seq_length = 512)
# corpus_ids = word_embedding_model.tokenize(data)['input_ids'].numpy()
# with open('run/data/vnPara/doc_ids_noPAD.txt', 'w', encoding = 'utf-8') as fout:
#         for doc_ids in corpus_ids:
#             for ids in doc_ids:
#                 fout.write(str(ids) + ' ') if ids != 0 else None
#             fout.write('\n')

new_data_id = []
with open('run/data/quora/doc_ids_noPAD.txt', 'r', encoding = 'utf-8') as fin:
    for line in fin:
        new_data_id.append(line.strip().split())

id2word = corpora.Dictionary(new_data_id)

corpus = [id2word.doc2bow(text) for text in new_data_id]

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=1,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

#Topic coherence
# coherence_score = CoherenceModel(model = lda_model, corpus = corpus, coherence='u_mass')
# print(coherence_score.get_coherence())

term_topic = lda_model.get_topics()
print(np.transpose(term_topic).shape)


val_list = list(id2word.values())
vocab = []
with open('run/vocab.txt','r') as fvocab:
    for line in fvocab:
        vocab.append(line.strip())

W = []
for i in range(len(vocab)):
    if str(i) in val_list:
        id_token = val_list.index(str(i))
        W.append(term_topic[:,id_token])
    else:
        W.append(np.zeros(term_topic.shape[0]))
print(np.array(W).shape)

np.save('run/data/quora/ids_topic/W_gensim_t1.npy', np.array(W))

