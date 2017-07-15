import pandas as pd
import numpy as np
import h5py
import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Flatten, Input, Dropout, MaxPooling1D, Convolution1D
from keras.layers import LSTM, Lambda, merge, Masking, SimpleRNN, GRU
from keras.layers import Embedding, TimeDistributed
from keras import backend as K
import keras.callbacks

from utils import clean_tweet, tokenize_tweet

np.random.seed(42)

df = pd.read_csv('../data/asonam/asonam.csv')
df['cleaned_tweet'] = df['text'].apply(clean_tweet)
df['tokenized_tweet'] =df['text'].apply(tokenize_tweet)

all_txt = ''
for tweet in df['cleaned_tweet'].values:
    all_txt += tweet

chars = set(all_txt)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 250

split_index = (2 * len(df)) / 3
df_train , df_test = df[:split_index], df[split_index:]

# Oversample training data with the clickbait class
#df_train_clickbait, df_train_no_clickbait = df_train[df_train['clickbait'] == 1], df_train[df_train['clickbait'] == 0]
#oversampled_df_train_clickbait = df_train_clickbait.sample(len(df_train_no_clickbait), replace=True, random_state=42)
#df_train = pd.concat([oversampled_df_train_clickbait, df_train_no_clickbait])

def binarize(x, sz=37):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))

def create_feature_matrix(docs):
    X = np.ones((len(docs), maxlen), dtype=np.int64) * -1

    for i, doc in enumerate(docs):
        for t, char in enumerate(doc):
            X[i, t] = char_indices[char]
        
    return X

X_train, X_test = create_feature_matrix(df_train['cleaned_tweet']), create_feature_matrix(df_test['cleaned_tweet'])
y_train, y_test = np.array(df_train['clickbait']), np.array(df_test['clickbait'])

X_train_lstm, X_test_lstm, y_train, y_test = 0, 0, 0, 0
with h5py.File('data_asonam.h5', 'r') as hf:
    print hf.keys()
    X_train_lstm = np.array(hf.get('X_train'))
    X_test_lstm = np.array(hf.get('X_test'))
    y_train = np.array(hf.get('y_train'))
    y_test = np.array(hf.get('y_test'))

ids = np.arange(len(X_train))
np.random.shuffle(ids)
X_train = X_train[ids]
y_train = y_train[ids]
X_train_lstm = X_train_lstm[ids]

dimsize = 300
sequence_size = 15
lstm_input = Input(shape=(sequence_size, dimsize))

def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 37

filter_length = [5, 3, 3]
nb_filter = [196, 196, 300]
pool_length = 2

in_sentence = Input(shape=(maxlen,), dtype='int64')
# binarize function creates a onehot encoding of each character index
embedded = Lambda(binarize, output_shape=binarize_outshape)(in_sentence)
# embedded: encodes sentence
for i in range(len(nb_filter)):
    embedded = Convolution1D(nb_filter=nb_filter[i],
                            filter_length=filter_length[i],
                            border_mode='valid',
                            activation='relu',
                            init='glorot_normal',
                            subsample_length=1)(embedded)

    embedded = Dropout(0.1)(embedded)
    embedded = MaxPooling1D(pool_length=pool_length)(embedded)

combined_features = merge([embedded, lstm_input], mode='concat', concat_axis=1)

forward_sent = SimpleRNN(128, return_sequences=False, dropout_W=0.2, dropout_U=0.2, consume_less='gpu')(combined_features)
backward_sent = SimpleRNN(128, return_sequences=False, dropout_W=0.2, dropout_U=0.2, consume_less='gpu', go_backwards=True)(combined_features)

sent_encode = merge([forward_sent, backward_sent], mode='concat', concat_axis=-1)
sent_encode = Dropout(0.3)(sent_encode)
output = Dense(1, activation='sigmoid')(sent_encode)

model = Model(input=[in_sentence,lstm_input], output=output)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'], class_mode='binary')
earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')

batch_size = 64
model.fit([X_train, X_train_lstm], y_train, batch_size=batch_size, nb_epoch=20,
          validation_split=0.1 , callbacks=[earlystop_cb])
score, acc = model.evaluate([X_test, X_test_lstm], y_test,
                            batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)

def predict_classes(model, X_test, X_test_lstm):
    proba = model.predict([X_test, X_test_lstm])
    if proba.shape[-1] > 1:
        return proba.argmax(axis=-1)
    else:
        return (proba > 0.5).astype('int32')

y_pred = predict_classes(model, X_test, X_test_lstm)
y_scores = model.predict([X_test, X_test_lstm])

with h5py.File('data_asonam_predictions.h5', 'w') as hf:
    hf.create_dataset('y_pred', data=y_pred)

# for idx, x in np.ndenumerate(y_test):
#     if y_test[idx[0]] != y_pred[idx[0]]:
#         print 'Truth: ' + str(y_test[idx[0]]) + ' Predicted: ' + str(y_pred[idx[0]])
#         print df['text'].ix[idx]

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
roc = roc_auc_score(y_test, y_scores)
print('ROC score:', roc)

metrics = classification_report(y_test, y_pred, digits=4)
print('Classification Report \n')
print metrics

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix \n')
print cm