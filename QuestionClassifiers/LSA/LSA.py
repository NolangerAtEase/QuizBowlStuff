#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 18:06:27 2019

@author: nolanesser
"""

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


