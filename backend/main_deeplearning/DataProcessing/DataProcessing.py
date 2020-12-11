import datetime
import pandas_datareader.data as pdr
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


class StockDataset():
    ### X_frame, y_frames = 시퀀스 랭스 정하는 변수.
    def __init__(self, symbol, x_frames, y_frames):

        ### StockDataset('005930',60.1)로 삼성전자 호출.

        self.symbol = symbol
        self.x_frames = x_frames  ## sequnce.
        self.y_frames = y_frames  ## 우리는 하루 예측이라서 항상 1

        self.data = pd.read_csv('../Dataset/' + symbol + '.csv')  # 데이터를 받기.
        del self.data['Unnamed: 0']
        print("number of MAN")
        print(self.data.isna().sum())

    ## sotfmax로 학습하기위해 y값 원핫 인코딩,
    def For_one_hot_train(self, train_rate):
        if 0 < train_rate and train_rate < 1:
            self.data_train = self.data[: int(self.data.shape[0] * train_rate)].copy()
            ###스케일 넘파이로 바뀜 !
            Scaler = MinMaxScaler()
            self.data_train = Scaler.fit_transform(self.data_train)
            ###트레인으로 짜름.

            self.X_data_train = []
            self.y_data_train = []

            #### one_hot 트레이닝 셋으로 만들기.
            for i in range(self.x_frames, self.data_train.shape[0]):

                self.X_data_train.append(self.data_train[i - self.x_frames: i])  ## 예를들어 x데이터 60일치 짜르기.

                if self.data_train[i, 3] >= self.data_train[i - 1, 3]:  ## 3번쨰 칼럼은 종가를 뜻함. y데이터는 종가보다 올랏으면 1 떨어졌으면 0
                    self.y_data_train.append(1)
                else:
                    self.y_data_train.append(0)

            self.X_data_train = np.array(self.X_data_train)
            self.y_data_train = np.array(self.y_data_train)

            ## X_data_train dim = (traing_set, sequence(X_frames) , colummns(x의 요소 ))
            return self.X_data_train, self.y_data_train

    def For_one_hot_test(self, test_rate):
        if 0 < test_rate and test_rate < 1:

            self.data_test = self.data[int(self.data.shape[0] * (1 - test_rate)):].copy()
            ###스케일
            Scaler = MinMaxScaler()
            self.data_test = Scaler.fit_transform(self.data_test)
            ###테스트으로 짜름.

            self.X_data_test = []

            self.y_data_test = []

            #### one_hot 테스트 셋으로 만들기.
            for i in range(self.x_frames, self.data_test.shape[0]):

                self.X_data_test.append(self.data_test[i - self.x_frames: i])  ## 예를들어 x데이터 60일치 짜르기.
                ## for문에서 마지막 포함 안하고 자르기에서 포함안해서 크기에서 -2 만큼 뒤에.

                if self.data_test[i, 3] >= self.data_test[i - 1, 3]:  ## 3번쨰 칼럼은 종가를 뜻함. y데이터는 종가보다 올랏으면 1 떨어졌으면 0
                    self.y_data_test.append(1)
                else:
                    self.y_data_test.append(0)

            self.X_data_test = np.array(self.X_data_test)
            self.y_data_test = np.array(self.y_data_test)

            return self.X_data_test, self.y_data_test

    ###하루예측 dim 맞춰주기. (# of training, # of sequence, # of x의 요소 )
    def For_tommorow(self):
        Scaler = MinMaxScaler()
        self.data_for_tommorow = self.data[self.data.shape[0] - self.x_frames:self.data.shape[0]]
        self.data_for_tommorow = Scaler.fit_transform(self.data_for_tommorow)
        self.data_for_tommorow = np.array(self.data_for_tommorow).reshape(1, self.x_frames, -1)
        return self.data_for_tommorow


### 로지스틱 정확도 함수 (label 이 두개일 때 . )
def accuracy(y_pred, y_test):
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1

    result = 0
    for i in range(0, y_pred.shape[0]):
        if y_pred[i] == y_test[i]:
            result += 1

    accuracy = (result / y_pred.shape[0]) * 100

    return accuracy



