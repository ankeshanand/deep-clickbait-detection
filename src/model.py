import pandas as pd
import numpy as np
import h5py

np.random.seed(42)
from sklearn.cross_validation import train_test_split
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Bidirectional
from keras.layers import LSTM, SimpleRNN, GRU
import keras.callbacks

from utils import tokenize_tweet
from constants import WORD2VEC_VECTORS_BIN

w2v = Word2Vec.load_word2vec_format(WORD2VEC_VECTORS_BIN, binary=True)
dimsize = 300
sequence_size = 15

def compute_training_matrix(Text):
    X = np.zeros((len(Text), sequence_size, dimsize))
    for idx, review in enumerate(Text):
        sequence = np.zeros((sequence_size, dimsize))
        tokens = review
        count = 0
        for token in tokens:
            if count == 15:
                break
            try:
                token = token.lower()
                sequence[count] = w2v[token]
                count += 1
            except:
                pass
        X[idx] = sequence

    return X

def create_training_test_matrices():
    # Load data
    df = pd.read_csv('../data/asonam/asonam.csv')

    split_index = (2 * len(df)) / 3
    df_train , df_test = df[:split_index], df[split_index:]

    # Oversample training data with the clickbait class
    #df_train_clickbait, df_train_no_clickbait = df_train[df_train['clickbait'] == 1], df_train[df_train['clickbait'] == 0]
    #oversampled_df_train_clickbait = df_train_clickbait.sample(len(df_train_no_clickbait), replace=True, random_state=42)
    #df_train = pd.concat([oversampled_df_train_clickbait, df_train_no_clickbait])

    Text_train, Text_test = df_train['text'].apply(tokenize_tweet), df_test['text'].apply(tokenize_tweet)
    y_train, y_test = df_train['clickbait'], df_test['clickbait']

    # create training and test data matrix
    X_train = compute_training_matrix(Text_train)
    X_test = compute_training_matrix(Text_test)

    # Save arrays to disk
    with h5py.File('data_asonam.h5', 'w') as hf:
        hf.create_dataset('X_train', data=X_train)
        hf.create_dataset('X_test', data=X_test)
        hf.create_dataset('y_train', data=y_train)
        hf.create_dataset('y_test', data=y_test)


#create_training_test_matrices()
X_train, X_test, y_train, y_test = 0, 0, 0, 0

# Reload data from disk
with h5py.File('data_asonam.h5', 'r') as hf:
    print hf.keys()
    X_train = np.array(hf.get('X_train'))
    X_test = np.array(hf.get('X_test'))
    y_train = np.array(hf.get('y_train'))
    y_test = np.array(hf.get('y_test'))

ids = np.arange(len(X_train))
np.random.shuffle(ids)
X_train = X_train[ids]
y_train = y_train[ids]

# build the keras LSTM model
model = Sequential()
model.add(Bidirectional(GRU(128, dropout_W=0.2, dropout_U=0.2), input_shape=(sequence_size, dimsize)))  # try using a GRU instead, for fun
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'], class_mode='binary')

batch_size = 64

print('Train...')
earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=20,
          validation_split=0.1, callbacks=[earlystop_cb])
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)

model.save_weights('lstm_model.hdf5')
with open('lstm_model.json', 'w') as f:
  f.write(model.to_json())

print('Test score:', score)
print('Test accuracy:', acc)

y_pred = model.predict_classes(X_test)
y_scores = model.predict_proba(X_test)

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
roc = roc_auc_score(y_test, y_scores)
print('ROC score:', roc)

metrics = classification_report(y_test, y_pred, digits=4)
print('Classification Report \n')
print metrics

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix \n')
print cm

with open('results.log', 'wa') as f:
    f.write(str(model.summary()))
    f.write('Roc Score:' + str(roc))
    f.write('Classification Report: \n')
    f.write(metrics)
    f.write('Confusion Matrix: \n')
    f.write(cm)
