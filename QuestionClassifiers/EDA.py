#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:45:15 2019

@author: nolanesser
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:30:34 2019

@author: nolanesser
"""

# Import the stuff that'll be needed
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from scipy.stats import rankdata
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout


# Import the hand-classified data
qs = pd.read_csv("data/questions_Reclassified.csv", engine = 'python')

####################
## Preprocessing  ##
####################


# Remove Incomplete questions and Trash questions
qs = qs[qs['IncompFlg'] != 1]
qs = qs[qs['Class'] != "Trash"]

# Look at split of question classes

sns.countplot(x = "Class", data = qs, order = qs['Class'].value_counts().index).set_title("Question Count by Class")

#Look at a few question texts:

# Set display column width so we can see the whole question
pd.options.display.max_colwidth = 2000
print(qs['Text'][:10])


# Do some regex-ing to clean up the text a bit
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))



def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('ftp', '') #Remove some common quizbowl-ese
#    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text


qs['CleanText'] = qs['Text'].apply(clean_text)


# Look at 50 or so mostfght take a while)
#
# split() returns list of all the words in the string




#Use this way:
allTxt = qs['Text'].to_string()
split_it = allTxt.split() 

# Pass the split_it list to instance of Counter class. 
c = Counter(split_it) 

# most_common() produces k frequently encountered 
# input values and their respective counts. 
most_occur = c.most_common(10000) 

#print(most_occur) 
c.clear()

#Plot Zipf's law for original Text
x, y = zip(*most_occur)
plt.scatter(np.log(range(1,10001)),np.log(y))

#Plot Zipf's law for cleaned Text
allCleanTxt = qs['CleanText'].to_string()
split_it = allCleanTxt.split() 
c = Counter(split_it) 
most_occur = c.most_common(1000) 
c.clear()
x, y = zip(*most_occur)
plt.scatter(np.log(range(1,1001)),np.log(y))

#Look at SVD to see if we see some clustering on regular tdf

vec = CountVectorizer()
text_counts = vec.fit_transform(qs['CleanText'][:10000])

svd = TruncatedSVD(n_components = 50, random_state = 1, n_iter = 20)
svd.fit(text_counts)


print(svd.explained_variance_ratio_)
print(svd.explained_variance_ratio_.sum())

svd_result = svd.fit_transform(text_counts)

df = pd.DataFrame()
df['svd-one'] = svd_result[:,0]
df['svd-two'] = svd_result[:,1]
df['svd-three'] = svd_result[:,2]

print('Explained variation per principal component: {}'.format(svd.explained_variance_ratio_))

sns.scatterplot(x = "svd-one", y= "svd-two", hue = qs['Class'][:1000], data = df)


tsne = TSNE(n_components = 2, verbose = 1)
tsne_results = tsne.fit_transform(df)


df2 = pd.DataFrame()
df2['tsne-2d-one'] = tsne_results[:,0]
df2['tsne-2d-two'] = tsne_results[:,1]

sns.scatterplot(x = "tsne-2d-one", y = "tsne-2d-two", hue = qs['Class'][:1000], data = df2)





# Do it for tf-idf
tfidf_vec = TfidfVectorizer()

tfidfMat = tfidf_vec.fit_transform(qs['CleanText'])
n_topics = 9

lsa_model = TruncatedSVD(n_components = n_topics)
lsa_topic_matrix = lsa_model.fit_transform(tfidfMat)






def get_keys(topic_matrix):
    '''
    returns an integer list of predicted topic 
    categories for a given topic matrix
    '''
    keys = topic_matrix.argmax(axis=1).tolist()
    return keys

def keys_to_counts(keys):
    '''
    returns a tuple of topic categories and their 
    accompanying magnitudes for a given list of keys
    '''
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)
    
lsa_keys = get_keys(lsa_topic_matrix)
lsa_categories, lsa_counts = keys_to_counts(lsa_keys)














# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 100000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 500
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(qs['CleanText'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(qs['CleanText'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(qs['Class']).values
print('Shape of label tensor:', Y.shape)


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)



model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(9, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


plot_history(history)




new_q = ['In this film, a shot of a man yelling “I could strangle her” dissolves to a shot of another character admiring his just-manicured hands, which are posed as if strangling an invisible neck. A character in this film gets carried away while demonstrating a murder method on Mrs. Cunningham and nearly kills her for real. This film’s climax takes place on an out-of-control (*) carousel at an amusement park where the villain had earlier killed Miriam, a murder that is shown in the reflection in a pair of glasses. A scene in this film cross-cuts between a tennis match and shots of the villain trying to retrieve a lighter inscribed “A to G.” For 10 points, name this Hitchcock film in which Bruno Anthony tries to convince Guy Haines to “swap murders” after they meet on the title conveyance.']

seq = tokenizer.texts_to_sequences(new_q)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
labels = ['FA', 'Geo', 'Hist', 'Lit', 'Myth', 'Philo', 'Religion', 'SS', 'Sci']
print(pred, labels[np.argmax(pred)])

pred = model.predict(X_test)

errs = 0
for i in range(0,4032):
    print(labels[np.argmax(pred[i])], labels[np.argmax(Y_test[i])])
    if labels[np.argmax(pred[i])] != labels[np.argmax(Y_test[i])]:
        errs += 1
        
print(errs)
from sklearn import metrics

matrix = metrics.confusion_matrix(Y_test.argmax(axis = 1), pred.argmax(axis = 1))
mlMatrix = metrics.multilabel_confusion_matrix(Y_test.argmax(axis = 1), pred.argmax(axis = 1))
cr = metrics.classification_report(Y_test.argmax(axis = 1), pred.argmax(axis = 1))
