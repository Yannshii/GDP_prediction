import warnings
from math import *
warnings.filterwarnings('ignore')

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

maxlen = 5
n_hidden = 300
nepoch = 100

maxlen = int(sys.argv[1])
n_hidden = int(sys.argv[2])
nepoch = int(sys.argv[3])
activation = str(sys.argv[4])


def make_dataset(low_data,  maxlen=25):

    data, target = [], []

    for i in range(len(low_data)-maxlen):
        data.append(low_data[i:i + maxlen])
        target.append(low_data[i + maxlen])

    re_data = np.array(data).reshape(len(data), maxlen, 1)
    re_target = np.array(target).reshape(len(data), 1)

    return re_data, re_target

path = "./data/5194/5194_2018.csv"
df = pd.read_csv(path, index_col="日付")
f = np.array(df["終値"])


#nepoch = 10**4


g, h = make_dataset(f, maxlen=10)
ttr = train_test_ratio = 0.9
ttsi = train_test_split_index = round(g.shape[0]*ttr)

tXm = train_X_mean = np.mean(g[:ttsi])
tXv = train_X_variace = sqrt(np.var(g[:ttsi]))


train_X, test_X = (g[:ttsi] - tXm)/sqrt(tXv), (g[ttsi:] - tXm)/sqrt(tXm)
train_Y, test_Y = (h[:ttsi] - tXm)/sqrt(tXv), (h[ttsi:] - tXm)/sqrt(tXm)

# tXm = train_X_mean = np.max(g[:ttsi])
# train_X, test_X = g[ttsi:]/tXm, g[ttsi:]/tXm
# train_Y, test_Y = h[ttsi:]/tXm, h[ttsi:]/tXm


length_of_sequence = g.shape[1]
in_out_neurons = 1
#n_hidden = 300


model = Sequential()
model.add(LSTM(n_hidden, \
    batch_input_shape=(None,
    length_of_sequence,
    in_out_neurons),
    return_sequences=False)
    )
model.summary()
model.add(Dense(in_out_neurons))
model.add(Activation(activation))
# model.add(Dense(round(n_hidden/10)))
# model.add(Activation("sigmoid"))
optimizer = Adam(lr=0.01)
model.compile(loss="mean_squared_error", optimizer=optimizer)
#early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20000)
early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=200000)

model.fit(train_X, train_Y,
          batch_size=300,
          epochs=nepoch,
          validation_split=0.1,
          callbacks=[early_stopping]
          )

predicted = model.predict(train_X)
print(predicted)
plt.figure()
plt.plot(range(maxlen,len(predicted)+maxlen), predicted*sqrt(tXv) + tXm, color="r", label="predict_data")
plt.plot(range(0, len(f)), f, color="b", label="row_data")
plt.legend()
plt.show()

test_pred = model.predict(test_X)
plt.figure()
plt.plot(range(0, len(test_Y)), test_Y*sqrt(tXv) + tXm, color="blue", label="row_data")
plt.plot(range(0, len(test_pred)), test_pred*sqrt(tXv) + tXm, color="r", label="prediction")
plt.legend()
plt.show()

print("rmse : ")
dif = sum(test_pred - test_Y)
print(sqrt(dif**2))
