import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, LSTM, TimeDistributed
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

def lstm_model(shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=(shape[1], 1), return_sequences=True))
    model.add(Dropout(rate=0.5))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    model.add(Dense(24,activation='relu'))
    model.compile(loss="mean_absolute_error", optimizer="adam",metrics=['mean_absolute_error'])
    model.summary()
    return model

def normalize(x, y):
    max_ = np.max(x)
    min_ = np.min(x)
    x = (x ) / (max_ )
    y = (y ) / (max_ )
    return [x, y, max_, min_]

def denormalize(x, max, min):
    return x * (max - min) + min

def func_zip(x):
    y = (np.log(x + 0.25) / 2) + 1
    return y

def func_unzip(y):
    x = np.exp((y - 1)* 2) - 0.25
    return x

def load(file_list, path):
    target_data = []
    for i in range(len(file_list)):
        data = pd.read_csv(path + file_list[i])
        target_data.append(data["consumption"].values)
    return target_data

def traindata_make(target_data):
    train_x = []
    train_y = []
    for data in target_data:
        # for d in range(len(data)):
        #     data[d] = func_zip(data[d])
        for i in range(len(data) // 24 - 7):
            x = data[i* 24: (i+ 7)* 24]
            y = data[(i+ 7)* 24: (i+ 8)* 24]
            # x, y, _, _ = normalize(x, y)
            train_x.append(x)
            train_y.append(y)
    return np.array(train_x), np.array(train_y)

def model_train(train_x, train_y):
    model = lstm_model(train_x.shape)
    callback = EarlyStopping(monitor="val_mean_absolute_error", patience=5, verbose=1, mode="auto")
    history = model.fit(train_x, train_y, epochs=30, batch_size=5, validation_split=0.1, callbacks=[callback], shuffle=True)
    return model

def consumption_pred(model, test_x):
    predict_y = []
    for x in test_x:
        predict_y.append(model.predict(np.array([x])))
    return predict_y

def generation_pred(week_gen):
    hours_gen = np.zeros(24)
    for d in range(7):
        hours_gen += week_gen[d * 24: (d + 1) * 24]
    peak = hours_gen[12] / 7
    hours_gen = hours_gen / np.max(hours_gen) * peak
    return hours_gen    

def trade_pred(consume, generate):
    trade_decision = []
    for i in range(24):
        if generate[i] < 0.2:
            trade_decision.append([i, "buy", 2.5256, consume[i]])
        elif generate[i] - consume[i] > 0.2:
            trade_decision.append([i, "sell", 1.2628, generate[i] - consume[i]])
    return trade_decision



import os
path = './training/'
model_name = "lstm_v5-nor.hdf5"
file_list = os.listdir(path)
target_data = load(file_list, path)

train_x, train_y = traindata_make(target_data)
# test_number = int(len(train_x) * 0.1)
# test_x = train_x[-test_number:]
# test_y = train_y[-test_number:]
# train_x = train_x[:-test_number]
# train_y = train_y[:-test_number]
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.1, random_state=1)

for i in range(len(train_x)):
    train_x[i], train_y[i], _, _ = normalize(train_x[i], train_y[i])


# model = model_train(train_x, train_y)
# model.save(model_name)
model = load_model(model_name)

predict_y = []
for i in range(len(test_x)):
    test_x[i], _, max, min = normalize(test_x[i], test_y[i])
    predict = consumption_pred(model, [test_x[i]])
    predict = denormalize(predict[0], max, min)
    predict_y.append(predict)
    # for j in range(len(test_y[i])):
    #     test_y[i][j] = func_unzip(test_y[i][j])

# predict_y = consumption_pred(model, test_x)
predict = np.reshape(predict_y,(len(test_y)*24))
test_y = np.reshape(test_y,(len(test_y)*24))
# predict_y = consumption_pred(model, [train_x[0]])

import matplotlib.pyplot as plt
plt.plot(test_y[:50*24])
plt.plot(predict[:50*24])
plt.legend(["test", "pred"], loc ="upper left")
plt.show()
