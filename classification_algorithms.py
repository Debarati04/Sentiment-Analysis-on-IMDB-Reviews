#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf 
from tensorflow import keras
import numpy as np
from keras.datasets import imdb
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


imdb_data = pd.read_csv('./IMDB Dataset.csv')
print(imdb_data.shape)
imdb_data.head(10)


# In[ ]:


imdb_data.info()


# In[ ]:


imdb_data.describe()


# See the average number of words of each sample

# In[ ]:


plt.figure(figsize=(10,6))
plt.hist([len(sample.split()) for sample in list(imdb_data['review']) ], 50)
plt.xlabel('number of words in review')
plt.ylabel('number of samples')
plt.title("review length distribution")


# In[ ]:


plt.figure(figsize=(5,5))
plt.hist([list(imdb_data['sentiment']) ], 5)
plt.xlabel('labels')
plt.ylabel('number of labels')
plt.title("sentiment distribution")


# In[ ]:


dict_w = {}
for line in imdb_data['review']:
    for word in line.split():
        if word in dict_w:
            dict_w[word] += 1
        else:
            dict_w[word] = 1
lists = sorted(dict_w.items(), key=lambda kv: kv[1], reverse=True) # sorted by value, return a list of tuples
x, y = zip(*lists[:30]) # unpack a list of pairs into two tuples, choose top 30
plt.figure(figsize=(40,20))
plt.bar(x, y)
plt.tick_params(axis='both', labelsize=25)
plt.title('top 30 frequence of words', size=40)
plt.show()


# Preliminary Preprocessing and Algorithms

# In[ ]:


import re
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import nltk
import string
from nltk.stem.snowball import SnowballStemmer

imdb_data = pd.read_csv('./IMDB Dataset.csv')

def clean_review(text):
    text = re.sub('<[^<]+?>', ' ', text)
    text = text.replace('\\"', '')
    text = text.replace('"', '')
    return text

imdb_data['cleaned_review'] = imdb_data['review'].apply(clean_review)
print(imdb_data.shape)
imdb_data.head(10)


# In[ ]:


nltk.download('stopwords')
from nltk.corpus import stopwords 
stop_words = stopwords
def clean_review(text):
    text = re.sub('<[^<]+?>', ' ', text)
    text = text.replace('\\"', '')
    text = text.replace('"', '')
    return text

def remove_stop_words(text):
  res = text.split(' ')
  final_res = []
  stop_words = set(stopwords.words('english'))
  for word in res:
    if word in stop_words:
      continue
    else:
      final_res.append(word)
  return " ".join(final_res)


# In[ ]:


import re
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import nltk
import string
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score




# In[ ]:


imdb_data = pd.read_csv('./IMDB Dataset.csv')
imdb_data['cleaned_review'] = imdb_data['review'].apply(clean_review)
imdb_data['cleaned_review'] = imdb_data['cleaned_review'].apply(remove_stop_words)
print(imdb_data.shape)
imdb_data.head(10)


# # Preprocessing

# split to train and test

# In[ ]:


#labeling the sentient data
lb=LabelBinarizer()
#transformed sentiment data
sentiment_label=lb.fit_transform(imdb_data['sentiment'])
print(sentiment_label.shape)

X_org_train, X_org_test, y_train, y_test = train_test_split(imdb_data['cleaned_review'], sentiment_label, test_size=0.2)


# In[ ]:


X_org_train.shape


# In[ ]:


def print_metrics(pred, label):
    print("accracy: ", accuracy_score(label, pred))
    print("precision: ", precision_score(label, pred))
    print("recal: ", recall_score(label, pred))
    print("f1_score: ", f1_score(label, pred))
    print("AUC: ", roc_auc_score(label, pred))


# ## BOW

# In[ ]:


vectorizer = CountVectorizer(lowercase=True)
vectorizer.fit(X_org_train)
X_train = vectorizer.transform(X_org_train)
X_test = vectorizer.transform(X_org_test)


# In[ ]:


X_train.shape


# ### Logistic Regression

# In[ ]:


lr = LogisticRegression(C=0.1)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
#lr_pred
#print(accuracy_score(y_test, lr.predict(X_test)))
print_metrics(lr_pred, y_test)


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
clf_forest = RandomForestClassifier(n_estimators=500, oob_score=False,max_depth=50)
clf_forest.set_params(max_features=5)
clf_forest= clf_forest.fit(X_train, y_train)
clf_forest_pred = clf_forest.predict(X_test)
#print(accuracy_score(y_test,clf_forest.predict(X_test)))
print_metrics(clf_forest_pred, y_test)


# ### Naive Bayes

# In[ ]:


clf_NB = MultinomialNB()
clf_NB.fit(X_train,y_train)
clf_NB_pred = clf_NB.predict(X_test)
#print(accuracy_score(clf.predict(X_test),y_test))
print_metrics(clf_NB_pred, y_test)


# ### KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=100)
neigh.fit(X_train,y_train)
neigh_pred = neigh.predict(X_test)
#print(accuracy_score(neigh.predict(X_test),y_test))
print_metrics(neigh_pred, y_test)


# ### Word Cloud

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
def cloud(data,backgroundcolor = 'white', width = 800, height = 600):
    wordcloud = WordCloud(stopwords = STOPWORDS, background_color = backgroundcolor,
                         width = width, height = height).generate(data)
    plt.figure(figsize = (15, 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

cloud(' '.join(imdb_data['cleaned_review']))


# In[ ]:





# ## Lexicon based

# ### tokenizer and get lexicon feature

# In[ ]:


from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer()
total_reviews = imdb_data['cleaned_review']
tokenizer.fit_on_texts(total_reviews)

X_train_tokens = tokenizer.texts_to_sequences(X_org_train)
X_test_tokens = tokenizer.texts_to_sequences(X_org_test)


# In[ ]:


vocab_size = len(tokenizer.word_index) + 1
lexiconscore = np.zeros(vocab_size)


# In[ ]:


X_train_tokens[0][2]


# Score of a word could be the number of positive reviews it has appeared minus the number of negative reviews it has appeared in.
# 
# if the sum of scores in a review is negative, then we predict his review as negative sentiment.
# else we predict it as positive sentiment.

# In[ ]:


from copy import deepcopy
class LexiconVectorizer():
    def __init__(self, lexiconscore, X):
        self.lexiconscore = lexiconscore
        
        #self.newmatrix = deepcopy(X)
    
    def fit(self, X, y):
        for index, line in enumerate(X):
            for word in line:
                if y[index] == 1:
                    self.lexiconscore[word]+=1
                else:
                    self.lexiconscore[word]-=1
        return self
    
    def transform(self, X):
        newmatrix = deepcopy(X)
        for i in range(len(X)):
            for j in range(len(X[i])):
                newmatrix[i][j] = self.lexiconscore[X[i][j]]
        return newmatrix


# In[ ]:


lexiconvectorizer = LexiconVectorizer(lexiconscore, X_train_tokens)
lexiconvectorizer.fit(X_train_tokens, y_train)
X_lexicon_train = lexiconvectorizer.transform(X_train_tokens)
X_lexicon_test = lexiconvectorizer.transform(X_test_tokens)


# In[ ]:


lexicon_pred = [1 if sum(X)>0 else 0 for X in X_lexicon_test]


# In[ ]:


print_metrics(lexicon_pred, y_test)


# ## TF-IDF

# In[ ]:


#Tfidf vectorizer
tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,2))
#transformed train reviews
tv.fit(X_org_train)
X_tfidf_train = tv.transform(X_org_train)
#transformed test reviews
X_tfidf_test = tv.transform(X_org_test)
print('Tfidf_train:',X_tfidf_train.shape)
print('Tfidf_test:',X_tfidf_test.shape)


# ### Logistical Regression

# In[ ]:


lr = LogisticRegression(C=1)
lr.fit(X_tfidf_train, y_train)
lr_pred = lr.predict(X_tfidf_test)
#lr_pred
#print(accuracy_score(y_test, lr.predict(X_test)))
print_metrics(lr_pred, y_test)


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
clf_forest = RandomForestClassifier(n_estimators=300, oob_score=False, max_depth=100)
clf_forest.set_params(max_features=50)
clf_forest= clf_forest.fit(X_tfidf_train, y_train)
clf_forest_pred = clf_forest.predict(X_tfidf_test)
#print(accuracy_score(y_test,clf_forest.predict(X_test)))
print_metrics(clf_forest_pred, y_test)


# ### Naive Bayes

# In[ ]:


clf_NB = MultinomialNB()
clf_NB.fit(X_tfidf_train,y_train)
clf_NB_pred = clf_NB.predict(X_tfidf_test)
#print(accuracy_score(clf.predict(X_test),y_test))
print_metrics(clf_NB_pred, y_test)


# ### KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(X_tfidf_train,y_train)
neigh_pred = neigh.predict(X_tfidf_test)
#print(accuracy_score(neigh.predict(X_test),y_test))
print_metrics(neigh_pred, y_test)


# In[ ]:





# ## Word Embedding - Word2Vector

# ### tokenizer and get embedding matrix

# In[ ]:


import gensim
EMBED_DIM = 100
w2v_model = gensim.models.Word2Vec(sentences=imdb_data['cleaned_review'],size=EMBED_DIM,window=5,workers=4,min_count=1)

tokenizer = Tokenizer()
total_reviews = imdb_data['cleaned_review']
tokenizer.fit_on_texts(total_reviews)

X_train_tokens = tokenizer.texts_to_sequences(X_org_train)
X_test_tokens = tokenizer.texts_to_sequences(X_org_test)
vocab_size = len(tokenizer.word_index) + 1
embedding_weights = np.zeros((vocab_size, EMBED_DIM))
for word, index in tokenizer.word_index.items():
    #embedding_vector = word2vec.get(word)
    try:
        embedding_weights[index] = w2v_model.wv.get_vector(word)
    except:
        pass 


# ### Mean embedding 

# now we can convert each word to an vector. so for a review(a sample in training set), we can average the vectors of the words to get the representation for that review.

# In[ ]:


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        #self.dim = len(word2vec.itervalues().next())
        self.dim = self.word2vec.shape[1]
        self.vocab_size = self.word2vec.shape[0]

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in range(self.vocab_size)]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])



# In[ ]:


meanEmbeddingV = MeanEmbeddingVectorizer(embedding_weights)

X_w2v_train = meanEmbeddingV.transform(X_train_tokens)
#transformed test reviews
X_w2v_test = meanEmbeddingV.transform(X_test_tokens)
print('Word2Vector_train:',X_w2v_train.shape)
print('Word2Vector_test:',X_w2v_test.shape)


# In[ ]:


X_w2v_train[0]


# #### logistical Regression

# In[ ]:


lr = LogisticRegression(C=1)
lr.fit(X_w2v_train, y_train)
lr_pred = lr.predict(X_w2v_test)

print_metrics(lr_pred, y_test)


# #### Random Forest

# In[ ]:


clf_forest = RandomForestClassifier(n_estimators=300, oob_score=False, max_depth=30)
clf_forest.set_params(max_features=20)
clf_forest= clf_forest.fit(X_w2v_train, y_train)
clf_forest_pred = clf_forest.predict(X_w2v_test)
#print(accuracy_score(y_test,clf_forest.predict(X_test)))
print_metrics(clf_forest_pred, y_test)


# #### Naive Bayes

# In[ ]:


def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

X_norm_w2v_train = scale(X_w2v_train, 0, 1)
X_norm_w2v_test = scale(X_w2v_test, 0, 1)
clf_NB = MultinomialNB()
clf_NB.fit(X_norm_w2v_train,y_train)
clf_NB_pred = clf_NB.predict(X_norm_w2v_test)
print_metrics(clf_NB_pred, y_test)


# #### KNN

# In[ ]:


neigh = KNeighborsClassifier(n_neighbors=20)
neigh.fit(X_w2v_train,y_train)
neigh_pred = neigh.predict(X_w2v_test)
print_metrics(neigh_pred, y_test)


# ### TF-IDF embedding

# Rather than just average the word vectors in an review, this time we average them weighted by TF-IDF

# In[ ]:


import collections

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = self.word2vec.shape[1]
        self.vocab_size = self.word2vec.shape[0]

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = collections.defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in range(self.vocab_size)] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


# In[ ]:


tfidfEmbeddingV = TfidfEmbeddingVectorizer(embedding_weights)
tfidfEmbeddingV.fit(X_train_tokens)
X_w2vtfidf_train = tfidfEmbeddingV.transform(X_train_tokens)
#transformed test reviews
X_w2vtfidf_test = tfidfEmbeddingV.transform(X_test_tokens)
print('Word2Vector_train:',X_w2vtfidf_train.shape)
print('Word2Vector_test:',X_w2vtfidf_test.shape)


# In[ ]:


X_w2vtfidf_train[0]


# #### Logistical Regression

# In[ ]:


lr = LogisticRegression()
lr.fit(X_w2vtfidf_train, y_train)
lr_pred = lr.predict(X_w2vtfidf_test)

print_metrics(lr_pred, y_test)


# #### Random Forest 

# In[ ]:


clf_forest = RandomForestClassifier(n_estimators=100, oob_score=False, max_depth=30)
clf_forest.set_params(max_features=20)
clf_forest= clf_forest.fit(X_w2vtfidf_train, y_train)
clf_forest_pred = clf_forest.predict(X_w2vtfidf_test)
#print(accuracy_score(y_test,clf_forest.predict(X_test)))
print_metrics(clf_forest_pred, y_test)


# #### Naive Bayes

# In[ ]:


def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

X_norm_w2vtfidf_train = scale(X_w2vtfidf_train, 0, 1)
X_norm_w2vtfidf_test = scale(X_w2vtfidf_test, 0, 1)
clf_NB = MultinomialNB()
clf_NB.fit(X_norm_w2vtfidf_train,y_train)
clf_NB_pred = clf_NB.predict(X_norm_w2vtfidf_test)
print_metrics(clf_NB_pred, y_test)


# #### KNN

# In[ ]:


neigh = KNeighborsClassifier(n_neighbors=20)
neigh.fit(X_w2vtfidf_train,y_train)
neigh_pred = neigh.predict(X_w2vtfidf_test)
print_metrics(neigh_pred, y_test)


# ## Word Embedding - GloVe

# ### get embedding matrix

# In[ ]:


# !wget http://nlp.stanford.edu/data/glove.6B.zip
# !unzip glove.6B.zip


# In[ ]:


with open("./glove/glove.6B.50d.txt", "rb") as lines:
    w2v_glove = {line.split()[0].decode("utf-8") : np.array(list(map(float, line.split()[1:])))
           for line in lines}
    


# In[ ]:


#vocab_size = len(tokenizer.word_index) + 1
embedding_glove_weights = np.zeros((vocab_size, 50))
for word, index in tokenizer.word_index.items():
    #embedding_vector = word2vec.get(word)
    try:
        embedding_glove_weights[index] = w2v_glove[word]
    except:
        pass 


# In[ ]:


embedding_glove_weights[20]


# ### Mean embedding

# In[ ]:


meanEmbeddingV2 = MeanEmbeddingVectorizer(embedding_glove_weights)

X_glove_train = meanEmbeddingV2.transform(X_train_tokens)
#transformed test reviews
X_glove_test = meanEmbeddingV2.transform(X_test_tokens)
print('GloVe_train:',X_glove_train.shape)
print('GloVe_test:',X_glove_test.shape)


# In[ ]:


X_glove_train[0]


# #### Logistical Regression

# In[ ]:


lr = LogisticRegression(C=1)
lr.fit(X_glove_train, y_train)
lr_pred = lr.predict(X_glove_test)

print_metrics(lr_pred, y_test)


# #### Random Forest

# In[ ]:


clf_forest = RandomForestClassifier(n_estimators=100, oob_score=False, max_depth=20)
clf_forest.set_params(max_features=10)
clf_forest= clf_forest.fit(X_glove_train, y_train)
clf_forest_pred = clf_forest.predict(X_glove_test)
print_metrics(clf_forest_pred, y_test)


# #### Naive Bayes

# In[ ]:


def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

X_norm_glove_train = scale(X_glove_train, 0, 1)
X_norm_glove_test = scale(X_glove_test, 0, 1)
clf_NB = MultinomialNB()
clf_NB.fit(X_norm_glove_train,y_train)
clf_NB_pred = clf_NB.predict(X_norm_glove_test)
print_metrics(clf_NB_pred, y_test)


# #### KNN

# In[ ]:


neigh = KNeighborsClassifier(n_neighbors=20)
neigh.fit(X_glove_train,y_train)
neigh_pred = neigh.predict(X_glove_test)
print_metrics(neigh_pred, y_test)


# ### TF-IDF embedding

# In[ ]:


tfidfEmbeddingV2 = TfidfEmbeddingVectorizer(embedding_glove_weights)
tfidfEmbeddingV2.fit(X_train_tokens)
X_glovetfidf_train = tfidfEmbeddingV2.transform(X_train_tokens)
#transformed test reviews
X_glovetfidf_test = tfidfEmbeddingV2.transform(X_test_tokens)
print('GloVe_train:',X_glovetfidf_train.shape)
print('GloVe_test:',X_glovetfidf_test.shape)


# In[ ]:


X_glovetfidf_train[0]


# #### Logistical Regression

# In[ ]:


lr = LogisticRegression()
lr.fit(X_glovetfidf_train, y_train)
lr_pred = lr.predict(X_glovetfidf_test)

print_metrics(lr_pred, y_test)


# #### Random Forest

# In[ ]:


clf_forest = RandomForestClassifier(n_estimators=100, oob_score=False, max_depth=20)
clf_forest.set_params(max_features=10)
clf_forest= clf_forest.fit(X_glovetfidf_train, y_train)
clf_forest_pred = clf_forest.predict(X_glovetfidf_test)
print_metrics(clf_forest_pred, y_test)


# #### Naive Bayes

# In[ ]:


def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

X_norm_glovetfidf_train = scale(X_glovetfidf_train, 0, 1)
X_norm_glovetfidf_test = scale(X_glovetfidf_test, 0, 1)
clf_NB = MultinomialNB()
clf_NB.fit(X_norm_glovetfidf_train, y_train)
clf_NB_pred = clf_NB.predict(X_norm_glovetfidf_test)
print_metrics(clf_NB_pred, y_test)


# #### KNN

# In[ ]:


neigh = KNeighborsClassifier(n_neighbors=20)
neigh.fit(X_glovetfidf_train,y_train)
neigh_pred = neigh.predict(X_glovetfidf_test)
print_metrics(neigh_pred, y_test)


# ## LSTM

# In[ ]:


from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D,Conv1D,LSTM
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


# with LSTM, we can use end to end, which means word embedding inherently appears in the first layer.

# ### tokenize

# In[ ]:



tokenizer = Tokenizer()
total_reviews = imdb_data['cleaned_review']
tokenizer.fit_on_texts(total_reviews)
#max_length = max([len(review.split()) for review in total_reviews])
max_length = 100
X_train_tokens = tokenizer.texts_to_sequences(X_org_train)
X_test_tokens = tokenizer.texts_to_sequences(X_org_test)
vocab_size = len(tokenizer.word_index) + 1
X_train_pad = pad_sequences(X_train_tokens,maxlen=max_length,padding="post")
X_test_pad = pad_sequences(X_test_tokens,maxlen=max_length,padding="post")
print("max_len:%d,vocab_size:%d"%(max_length,vocab_size))


# In[ ]:


X_train_pad.shape


# In[ ]:


X_train_pad[0]


# ### embeddings

# In[ ]:


#import gensim
#EMBED_DIM = 100
#w2v_model = gensim.models.Word2Vec(sentences=imdb_data['cleaned_review'],size=EMBED_DIM,window=5,workers=4,min_count=1)


# In[ ]:


# embedding_weights = np.zeros((vocab_size, EMBED_DIM))
# for word, index in tokenizer.word_index.items():
#     #embedding_vector = word2vec.get(word)
#     try:
#         embedding_weights[index] = w2v_model.wv.get_vector(word)
#     except:
#         pass 


# ### LSTM model

# In[ ]:


#max_features = 20000
model = Sequential()
#embedding_layer = Embedding(vocab_size, EMBED_DIM, weights=[embedding_weights], input_length=max_length , trainable=False)
model.add(Embedding(vocab_size, 128))
#model.add(embedding_layer)
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


# In[ ]:


model.summary()


# **!!!the following cell will takes 10 min time on laptop!!!**

# In[ ]:


history =  model.fit(X_train_pad, y_train,  batch_size=32, epochs=6, validation_split=0.2,verbose=1)


# In[ ]:


def plot_lstm(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()


# In[ ]:


plot_lstm(history)


# ### test lstm

# In[ ]:


score = model.evaluate(X_test_pad, y_test, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[ ]:




