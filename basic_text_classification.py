#!/usr/bin/env python
# coding: utf-8

# In[14]:


import tensorflow as tf
from tensorflow import keras
import numpy as np


# In[23]:


imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[1])


# In[24]:


word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print (decode_review(train_data[1]))


# In[17]:


train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=256)


# In[18]:


vocab_size = 10000

model = keras.Sequential([keras.layers.Embedding(vocab_size, 16), keras.layers.GlobalAveragePooling1D(), keras.layers.Dense(16, activation=tf.nn.relu), keras.layers.Dense(1, activation=tf.nn.sigmoid)])


# In[19]:


model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])


# In[20]:


x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


# In[21]:


history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


# In[22]:


results = model.evaluate(test_data, test_labels)
print(results)


# In[ ]:




