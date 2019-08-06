from util import get_num_lines, get_pos2idx_idx2pos, index_sequence, get_vocab, embed_indexed_sequence, \
    get_word2idx_idx2word, get_embedding_matrix, write_predictions, get_performance_VUAverb_val, \
    get_performance_VUAverb_test, get_performance_VUA_test
from util import TextDatasetWithGloveElmoSuffix as TextDataset
from util import evaluate
from model_rnn_mhca import RNNSequenceModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import csv
import h5py
import pickle
import ast
import matplotlib.pyplot as plt
import random
import numpy as np

# Modified based on Gao Ge https://github.com/gao-g/metaphor-in-context

print("PyTorch version:")
print(torch.__version__)
print("GPU Detected:")
print(torch.cuda.is_available())
using_GPU = True

random.seed(4)
np.random.seed(4)
torch.manual_seed(4)
torch.cuda.manual_seed(4)
torch.backends.cudnn.deterministic = True


"""
1. Data pre-processing
"""
'''
1.1 VUA
get raw dataset as a list:
  Each element is a triple:
    a sentence: string
    a list of labels: 
    a list of pos: 

'''
pos_set = set()
raw_train_vua = []
with open('../data/VUAsequence/VUA_seq_formatted_train.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        pos_seq = ast.literal_eval(line[4])
        label_seq = ast.literal_eval(line[3])
        assert (len(pos_seq) == len(label_seq))
        assert (len(line[2].split()) == len(pos_seq))
        raw_train_vua.append([line[2], label_seq, pos_seq])
        pos_set.update(pos_seq)

raw_val_vua = []
with open('../data/VUAsequence/VUA_seq_formatted_val.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        pos_seq = ast.literal_eval(line[4])
        label_seq = ast.literal_eval(line[3])
        assert (len(pos_seq) == len(label_seq))
        assert (len(line[2].split()) == len(pos_seq))
        raw_val_vua.append([line[2], label_seq, pos_seq])
        pos_set.update(pos_seq)

# embed the pos tags
pos2idx, idx2pos = get_pos2idx_idx2pos(pos_set)

for i in range(len(raw_train_vua)):
    raw_train_vua[i][2] = index_sequence(pos2idx, raw_train_vua[i][2])
for i in range(len(raw_val_vua)):
    raw_val_vua[i][2] = index_sequence(pos2idx, raw_val_vua[i][2])
print('size of training set, validation set: ', len(raw_train_vua), len(raw_val_vua))


"""
2. Data preparation
"""
'''
2. 1
get vocabulary and glove embeddings in raw dataset 
'''
# vocab is a set of words
vocab = get_vocab(raw_train_vua)
# two dictionaries. <PAD>: 0, <UNK>: 1
word2idx, idx2word = get_word2idx_idx2word(vocab)
# glove_embeddings a nn.Embeddings
glove_embeddings = get_embedding_matrix(word2idx, idx2word, normalization=False)
elmos_train_vua = h5py.File('../elmo/VUA_train.hdf5', 'r')
elmos_val_vua = h5py.File('../elmo/VUA_val.hdf5', 'r')
bert_train_vua = None
bert_val_vua = None
# no suffix embeddings for sequence labeling
suffix_embeddings = None
#suffix_embeddings = nn.Embedding(15, 50)

'''
2. 2
embed the datasets
'''
# raw_train_vua: sentence, label_seq, pos_seq
# embedded_train_vua: embedded_sentence, pos, labels
embedded_train_vua = [[embed_indexed_sequence(example[0], example[2], word2idx,
                                      glove_embeddings, elmos_train_vua, bert_train_vua, suffix_embeddings),
                       example[2], example[1]]
                      for example in raw_train_vua]
embedded_val_vua = [[embed_indexed_sequence(example[0], example[2], word2idx,
                                    glove_embeddings, elmos_val_vua, bert_val_vua, suffix_embeddings),
                     example[2], example[1]]
                    for example in raw_val_vua]

'''
2. 3
set up Dataloader for batching
'''
train_dataset_vua = TextDataset([example[0] for example in embedded_train_vua],
                                [example[1] for example in embedded_train_vua],
                                [example[2] for example in embedded_train_vua])
val_dataset_vua = TextDataset([example[0] for example in embedded_val_vua],
                              [example[1] for example in embedded_val_vua],
                              [example[2] for example in embedded_val_vua])

# Data-related hyperparameters
batch_size = 2
train_dataloader_vua = DataLoader(dataset=train_dataset_vua, batch_size=batch_size, shuffle=True,
                              collate_fn=TextDataset.collate_fn)
val_dataloader_vua = DataLoader(dataset=val_dataset_vua, batch_size=batch_size,
                            collate_fn=TextDataset.collate_fn)

"""
3. Model training
"""
'''
3. 1 
set up model, loss criterion, optimizer
'''
#RNN_HG model
RNNseq_model = RNNSequenceModel(num_classes=2, embedding_dim=1324, hidden_size=256, num_layers=1, bidir=True,
                                dropout1=0.5, dropout2=0, dropout3=0.2)

# Move the model to the GPU if available
if using_GPU:
    RNNseq_model = RNNseq_model.cuda()

# Set up criterion for calculating loss
class_weights = torch.FloatTensor([1, 2]).cuda()
loss_criterion = nn.NLLLoss(weight=class_weights)
rnn_optimizer = optim.SGD(RNNseq_model.parameters(), lr=0.2)

# Number of epochs (passes through the dataset) to train the model for.
num_epochs = 29

'''
3. 2
train model
'''
train_loss = []
val_loss = []
performance_matrix = None
val_f1s = []
train_f1s = []
# A counter for the number of gradient updates
num_iter = 0
comparable = []
for epoch in range(num_epochs):
    print("Starting epoch {}".format(epoch + 1))
    for (__, example_text, example_lengths, labels) in train_dataloader_vua:
        example_text = Variable(example_text)
        example_lengths = Variable(example_lengths)
        labels = Variable(labels)
        if using_GPU:
            example_text = example_text.cuda()
            example_lengths = example_lengths.cuda()
            labels = labels.cuda()
        predicted = RNNseq_model(example_text, example_lengths)
        batch_loss = loss_criterion(predicted.view(-1, 2), labels.view(-1))
        rnn_optimizer.zero_grad()
        batch_loss.backward()
        rnn_optimizer.step()
        num_iter += 1
        # Calculate validation and training set loss and accuracy every 3000 gradient updates
        if num_iter % 3000 == 0:
            avg_eval_loss, performance_matrix = evaluate(idx2pos, val_dataloader_vua, RNNseq_model,
                                                         loss_criterion, using_GPU)
            val_loss.append(avg_eval_loss)
            val_f1s.append(performance_matrix[:, 2])
            print("Iteration {}. Validation Loss {}.".format(num_iter, avg_eval_loss))

# Additional training
rnn_optimizer = optim.SGD(RNNseq_model.parameters(), lr=0.1)
additional_epochs = 9
for epoch in range(additional_epochs):
    print("Starting epoch {}".format(epoch + 1))
    for (__, example_text, example_lengths, labels) in train_dataloader_vua:
        example_text = Variable(example_text)
        example_lengths = Variable(example_lengths)
        labels = Variable(labels)
        if using_GPU:
            example_text = example_text.cuda()
            example_lengths = example_lengths.cuda()
            labels = labels.cuda()
        predicted = RNNseq_model(example_text, example_lengths)
        batch_loss = loss_criterion(predicted.view(-1, 2), labels.view(-1))
        rnn_optimizer.zero_grad()
        batch_loss.backward()
        rnn_optimizer.step()
        num_iter += 1
        # Calculate validation and training set loss and accuracy every 3000 gradient updates
        if num_iter % 3000 == 0:
            avg_eval_loss, performance_matrix = evaluate(idx2pos, val_dataloader_vua, RNNseq_model,
                                                         loss_criterion, using_GPU)
            val_loss.append(avg_eval_loss)
            val_f1s.append(performance_matrix[:, 2])
            print("Iteration {}. Validation Loss {}.".format(num_iter, avg_eval_loss))

raw_test_vua = []
with open('../data/VUAsequence/VUA_seq_formatted_test.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        # txt_id	sen_ix	sentence	label_seq	pos_seq	labeled_sentence	genre
        pos_seq = ast.literal_eval(line[4])
        label_seq = ast.literal_eval(line[3])
        assert(len(pos_seq) == len(label_seq))
        assert(len(line[2].split()) == len(pos_seq))
        raw_test_vua.append([line[2], label_seq, pos_seq])
print('number of examples(sentences) for test_set ', len(raw_test_vua))

for i in range(len(raw_test_vua)):
    raw_test_vua[i][2] = index_sequence(pos2idx, raw_test_vua[i][2])

elmos_test_vua = h5py.File('../elmo/VUA_test.hdf5', 'r')
bert_test_vua = None
embedded_test_vua = [[embed_indexed_sequence(example[0], example[2], word2idx,
                                      glove_embeddings, elmos_test_vua, bert_test_vua, suffix_embeddings),
                       example[2], example[1]]
                      for example in raw_test_vua]
test_dataset_vua = TextDataset([example[0] for example in embedded_test_vua],
                              [example[1] for example in embedded_test_vua],
                              [example[2] for example in embedded_test_vua])

# Set up a DataLoader for the test dataset
test_dataloader_vua = DataLoader(dataset=test_dataset_vua, batch_size=batch_size,
                              collate_fn=TextDataset.collate_fn)

print("Tagging model performance on VUA test set by POS tags: regardless of genres")
avg_eval_loss, performance_matrix = evaluate(idx2pos, test_dataloader_vua, RNNseq_model, loss_criterion, using_GPU)

"""
write the test prediction on the VUA-verb to a file: sequence prediction
read and extract to get a comparabel performance on VUA-verb test set.
"""
def get_comparable_performance_test():
    result = write_predictions(raw_test_vua, test_dataloader_vua, RNNseq_model, using_GPU, '../data/VUAsequence/VUA_seq_formatted_test.csv')
    f = open('../predictions/vua_seq_test_predictions_LSTMsequence_vua.csv', 'w')
    writer = csv.writer(f)
    writer.writerows(result)
    f.close()

    get_performance_VUAverb_test()
    get_performance_VUA_test()

get_comparable_performance_test()
