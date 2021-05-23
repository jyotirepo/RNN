# -*- coding: utf-8 -*-
"""
Created on Sun May 23 22:40:27 2021

@author: jysethy
"""

## Fake news classifier using LSTM - RNN

## Importing required librarries

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM,Dense

## Reading Dataset

df = pd.read_csv('train.csv')


## Droping null values using dropna

df = df.dropna()

##spliting dataset into X and y 

X = df.drop('label', axis=1)

y = df['label']

## Assigning Vobulary size to use word embeddings

voc_size = 5000

## One_hot representation

message = X.copy()


message.reset_index(inplace=True)


import nltk
import re
from nltk.corpus import stopwords
from tensorflow.keras.layers import Dropout

## Data Pre-processing

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(message)):
    print(i)
    review = re.sub('[^a-zA-Z]', ' ', message['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
onehot_repr=[one_hot(words,voc_size)for words in corpus] 


# Inlude zero padding using pad_squence

sent_size = 20

embedded_docs = pad_sequences(onehot_repr,padding='pre',maxlen=sent_size)

## Creaating the model

embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_size))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())



import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)

## Train-test split


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)


## Model Training.

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)


## Model Evaluation

#y_pred = model.predict(X_test)
y_pred = np.argmax(model.predict(X_test) < 0.5).astype("int32")

## Confusion Matrix
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
