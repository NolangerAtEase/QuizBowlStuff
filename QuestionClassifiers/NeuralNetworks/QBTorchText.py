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
















#qbSmall.to_csv('qbTextAndLabels.csv')

tokenize = lambda x: x.split()
txt = data.Field(sequential=True, tokenize=tokenize, lower=True)
lab = data.Field(sequential = False, use_vocab = True)


qbtab = data.TabularDataset('qbTextAndLabels.csv', format= 'csv', fields = 
                            [('id', data.Field()),
                             ('Class', data.Field()),
                             ('CleanText', data.Field())])


train, test = data.TabularDataset.split(qbtab)


txt.build_vocab(train)


train['CleanText'].build_vocab()



















#Test/Train split
X = qb['Text']
y = qb['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed, test_size = 0.2)

print(len(X_train), len(X_test), len(y_train), len(y_test))





















