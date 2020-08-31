#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:06:38 2020

@author: nolanesser
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

import torch
from torchtext import data
from torchtext import vocab
import torch.nn as nn

import re
from nltk.corpus import stopwords

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

data_dir = 'data/questions_reclassified.csv'

qb = pd.read_csv(data_dir, header = 0)# Dataset is now stored in a Pandas 
# Reset indices to make it easier to train/test split later
qb = qb.reset_index()

qb.tail()



#Filter out Trash questions and incomplete questions
qb = qb[qb['IncompFlg'] != 1]
qb = qb[qb['Class'] != "Trash"]
qb = qb.reset_index()
qb.tail()



REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    text = text.lower()    
    text = BAD_SYMBOLS_RE.sub('', text)
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = text.replace("ftp", '')
    text = text.replace("for ten points", '')
    text = text.replace("for 10 points", "")
    text = text.replace("|||", '')

    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text



qb['CleanText'] = qb['Text'].apply(clean_text)

qbSmall = qb[['Class', 'CleanText']]


train_df, test_df = train_test_split(qbSmall, test_size = 0.2)

#import spacy
#nlp = spacy.load('en_core_web_sm')

txt = data.Field( include_lengths = True)
labs = data.LabelField()



class DataFrameDataset(data.Dataset):
    
    def __init__(self, df, fields, is_test = False, **kwargs):
        examples = []
        for i, row in df.iterrows():
            label = row.Class if not is_test else None
            text = row.CleanText
            examples.append(data.Example.fromlist([text, label], fields))
            
        super().__init__(examples, fields, **kwargs)
        
    @staticmethod 
    def sort_key(ex):
        return len(ex.text)
    
    @classmethod
    def splits(cls, fields, train_df, test_df = None, **kwargs):
        train_data, test_data = (None, None)
        data_field = fields
        
        if train_df is not None:
            train_data = cls(train_df.copy(), data_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), data_field, True, **kwargs)
            
        return tuple(d for d in (train_data, test_data) if d is not None)
    
    
fields = [('text', txt), ('label', labs)]

train_ds, test_ds = DataFrameDataset.splits(fields, train_df = train_df, test_df = test_df)



print(vars(train_ds[0]))
print(type(train_ds[0]))

max_vocab_size = 25000

txt.build_vocab(train_ds, max_size = max_vocab_size, vectors = vocab.GloVe(), unk_init = torch.Tensor.zero_)
labs.build_vocab(train_ds)


batch_size = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_it, test_it = data.BucketIterator.splits((train_ds, test_ds), batch_size = batch_size, 
                                               sort_within_batch = True, device = device)




# Hyperparameters for LSTM
num_epochs = 25
learning_rate = 0.001

input_dim = len(txt.vocab)
embedding_dim = 300
hidden_dim = 256
output_dim = 9
n_layers = 2
bidirectional = True
dropout = 0.2
pad_idx = txt.vocab.stoi[txt.pad_token]



class LSTM_net(nn.Module):
    def __init__(self, vocab_size, embed_dim, hid_dim, out_dim, num_layers, bidirectional, drop, pad_index):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx = pad_index)
        
        self.rnn = nn.LSTM(embed_dim, hid_dim, num_layers=num_layers, bidirectional = bidirectional, dropout = dropout)
        
        self.fc1 = nn.Linear(hid_dim*2, hid_dim)
        self.fc2 = nn.Linear(hid_dim, 9)
        self.dropout = nn.Dropout(drop)
        
    def forward (self, text, text_lengths):
        
        embedded = self.embedding(text)
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)


        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        output = self.fc1(hidden)
        output = self.dropout(self.fc2(output))
        
        return output



model = LSTM_net(input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional,
                 dropout, pad_idx)






pretrained_embeddings = txt.vocab.vectors

print(pretrained_embeddings.shape)
model.embedding.weight.data.copy_(pretrained_embeddings)


model.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)
print(model.embedding.weight.data)


model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)#, weight_decay = 0.0001)

def accuracy(preds, y):
    
    #predicted = torch.max(preds, 1)
    _, predicted = torch.max(preds, 1)
    #print("input prediction" + preds)
    #print(y)
    #print(predicted)
    #print(predicted.shape)
    #print(predicted == y)
    
    correct = (predicted == y)
    acc = correct.sum()/ len(correct)
    return acc




#Training function
def train(model, iterator):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        text, text_lengths = batch.text
        
        optimizer.zero_grad()
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = accuracy(predictions, batch.label)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss/len(iterator), epoch_acc/ len(iterator)


def evaluate(model, iterator):
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for batch in iterator():
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            acc = accuracy(predictions, batch.labe)
            
            epoch_acc += acc.item()
            
    return epoch_acc/ len(iterator)


t = time.time()
loss = []
acc = []
test_acc = []

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_it)
    testing_acc = evaluate(model, test_it)
    
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Test Acc: {testing_acc*100:.2f}%')

    loss.append(train_loss)
    acc.append(train_acc)
    test_acc.append(testing_acc)
    
print(f'time:{time.time()-t:.3f}')
