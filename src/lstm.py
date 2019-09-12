# Import of modules
## python 3.7.1

import warnings
from math import *
import sys
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Definition of CLI inputs
maxlen = int(sys.argv[1])
n_hidden = int(sys.argv[2])
nepoch = int(sys.argv[3])
activation = str(sys.argv[4])
pic_path1 = str(sys.argv[5])
pic_path2 = str(sys.argv[6])

# maxlen = 5
# n_hidden = 100
# nepoch = 1000
# activation = "linear"


# Definition of functions to data arrangement

def build_timeseries(mat, y_col_index, TIME_STEPS=maxlen):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))

    for i in tqdm_notebook(range(dim_0)):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i, y_col_index]

    print("length of time-series i/o",x.shape,y.shape)
    return x, y


def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat



# Arrangement of the data
## Reading GDP data

path = "./data/GDPC1.csv"
df = pd.read_csv(path, index_col="DATE")
f = np.array(df["GDPC1"])

## Train - Test Split

df_train, df_test = train_test_split(df, train_size=0.8, test_size=0.2, shuffle=False)
x = df_train[["GDPC1"]].values

## Scaling
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x)
x_test = min_max_scaler.transform(df_test[["GDPC1"]].values)


## trimming for BATCH_SIZE and length of time series

x_t, y_t = build_timeseries(x_train, 0)
x_temp, y_temp = build_timeseries(x_test, 0)
BATCH_SIZE = round((y_temp.shape[0]%10*10) /2)
BATCH_SIZE = round((y_temp.shape[0]/2))
#BATCH_SIZE = (y_temp.shape[0]%20*20)
x_t = trim_dataset(x_t, BATCH_SIZE)
y_t = trim_dataset(y_t, BATCH_SIZE)
x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE),2)
y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE),2)


# Model construction

lstm_model = Sequential()
lstm_model.add(LSTM(100, batch_input_shape=(BATCH_SIZE, maxlen, x_t.shape[2]),\
    dropout=0.0, recurrent_dropout=0.0, stateful=True, kernel_initializer='random_uniform'))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(n_hidden, activation=activation))
lstm_model.add(Dense(1,activation='sigmoid'))
optimizer = RMSprop(lr=0.001)
lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)


# Fitting
history = lstm_model.fit(x_t, y_t, epochs=nepoch, verbose=2, batch_size=BATCH_SIZE,
                    shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                    trim_dataset(y_val, BATCH_SIZE)))


# plot

predicted = lstm_model.predict(x_t, batch_size=BATCH_SIZE)
predicted = min_max_scaler.inverse_transform(predicted)
y_t_inv = min_max_scaler.inverse_transform(y_t.reshape(len(y_t), 1))
plt.figure(figsize=(15,9))
plt.plot(range(maxlen,len(predicted)+maxlen), predicted, color="r", label="fitting")
plt.plot(range(0, len(y_t)), y_t_inv, color="b", label="row_data")
plt.xticks(range(0, len(y_t)), df.index[:len(y_t)])
plt.tick_params(labelsize=2, rotation=90)
plt.title("Fitting accuracy for training data set")
plt.legend()
plt.savefig(pic_path1)
#plt.show()


test_pred = lstm_model.predict(x_test_t)
y_test_inv = min_max_scaler.inverse_transform(y_test_t.reshape(len(y_test_t), 1))
test_pred_inv = min_max_scaler.inverse_transform(test_pred)
plt.figure(figsize=(15,9))
plt.plot(range(0, len(y_test_t)), y_test_inv, color="blue", label="row_data")
plt.plot(range(0, len(y_test_t)), test_pred_inv, color="r", label="prediction")
plt.xticks(range(0, len(y_test_t)), df.index[len(y_t):])
plt.tick_params(labelsize=6, rotation=90)
plt.title("Prediction accuracy for test data set")
plt.legend()
plt.savefig(pic_path2)
#plt.show()


plt.figure(figsize=(15,9))
plt.plot(lstm_model.history.history["loss"])
plt.plot(lstm_model.history.history["val_loss"])
plt.title("model loss")
plt.xlabel("loss")
plt.legend(["train", "validation"])
plt.savefig("result/pictures/example_of_model_loss.png")

print("rmse : ")
dif = sum(test_pred_inv - y_test_inv)
print(sqrt(dif**2))
