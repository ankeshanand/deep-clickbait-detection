import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU

from utils import review_to_words
from constants import WORD2VEC_VECTORS_BIN

df = pd.read_csv('../data/data.csv')
df['cleaned_text'] = df['text'].apply(review_to_words)
Text = df['cleaned_text']
y = df['clickbait']
del df

dimsize = 300

# train word2vec model
w2v = Word2Vec.load_word2vec_format(WORD2VEC_VECTORS_BIN, binary=True)

# create training and test data matrix
sequence_size = 15
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

nsamples, nx, ny = X.shape
X_reshaped = np.reshape(X, (nsamples, nx*ny))

from imblearn.over_sampling import SMOTE
sm = SMOTE(kind='regular')
X_reshaped, y = sm.fit_sample(X_reshaped, y)
print y.shape

X = np.reshape(X_reshaped, (X_reshaped.shape[0], nx, ny))
print X.shape

# Split into training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# build the keras LSTM model
model = Sequential()
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2, input_shape=(sequence_size, dimsize)))  # try using a GRU instead, for fun
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'], class_mode='binary')

batch_size = 64

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

y_pred = model.predict_classes(X_test)


from sklearn.metrics import roc_auc_score
roc = roc_auc_score(y_test, y_pred)
print('Roc Score:', roc)

from sklearn.metrics import confusion_matrix
print confusion_matrix(y_test, y_pred)
