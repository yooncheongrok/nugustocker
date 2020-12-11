import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense , LSTM, Dropout

import numpy as np
import pandas as pd

import seaborn as sns
from DataProcessing.DataProcessing import *
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader.data as pdr
import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import json

from DataProcessing.DataProcessing import *





dataset = StockDataset('005930', 60, 1)  ##005930 삼성전자

X_train, y_train = dataset.For_one_hot_train(0.8)
X_test, y_test = dataset.For_one_hot_test(0.2)
X_for_tommorow = dataset.For_tommorow()

##모델.
regression = Sequential()
regression.add(LSTM(units=60, activation='relu', return_sequences=True,
                    input_shape=(X_train.shape[1], X_train.shape[2])))
regression.add(Dropout(0.1))

### 마지막 아웃풋 many to one 모델.
regression.add(LSTM(units=60, activation='relu'))  ###마지막 아웃풋이라 시퀀스 리턴이 필요가없음. 아웃풋 1나와야됨.
regression.add(Dropout(0.1))

### 마지막 AL을 다시 fully nn 에 연결. 로지스틱이므로 마지막은 시그모이드로 출력
regression.add(Dense(units=1))
regression.add(layers.Activation(activation='sigmoid'))

regression.summary()  ### [batche, 인풋, 아웃풋]

## amdam으로 초적화 하고 loss 함수는 label 이 2개임으로 binary_crossentorpy를 이용하였음


regression.compile(optimizer='adam', loss='binary_crossentropy')

### training 시작.

regression.fit(X_train, y_train, epochs=10, batch_size=32, shuffle=True)

regression.save("./005930.h5")

##예측값.
y_pred_train = regression.predict(X_train)
y_pred_test = regression.predict(X_test)
y_pred_tommorow = regression.predict(X_for_tommorow)
up = y_pred_tommorow[0, 0]
down = 1 - y_pred_tommorow[0, 0]

# In[8]:


probability = {"probability": {"up": str(up), "down": str(down)}}

# In[9]:


##정확도 계산과 내일 몇 % 오를지 측정.
print("train accuracy : %s %%" % accuracy(y_pred_train, y_train))
print("test accuracy : %s %%" % accuracy(y_pred_test, y_test))
print("up : %s %%, down : %s %%" % (up * 100, down * 100))  ## 종목 내일의 확률

# In[10]:


probability_json = json.dumps(str(probability), indent=4)