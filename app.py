import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def trainingWindows(df, ref_day=1, predict_day=1):
    X_train, Y_train = [], []
    for i in range(df.shape[0] - predict_day - ref_day):
        X_train.append(np.array(df.iloc[i:i + ref_day]))
        Y_train.append(np.array(df.iloc[i + ref_day:i + ref_day + predict_day]['open']))
    return np.array(X_train), np.array(Y_train)


def modelBuilding(shape):
    model = Sequential()
    model.add(LSTM(units = 256, kernel_initializer = 'glorot_normal', return_sequences = True, input_shape = (shape[1], shape[2])))

    model.add(LSTM(units = 256, kernel_initializer = 'glorot_normal', return_sequences = True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    
    model.add(Dense(5,activation='linear'))
    model.add(Dense(1,activation='linear'))
    
    model.compile(loss="mean_absolute_error", optimizer="adam",metrics=['mean_absolute_error'])
    
    model.summary()
    
    return model


class Trader():
    
    def __init__(self, args):
        self.Model = None
        self.TrainingFName = args.training
        self.TestingFName = args.testing
        self.Training = None
        self.Testing = None
        self.TPredictData = None
        self.Scaler = MinMaxScaler(feature_range=(0, 1))
        self.WindowSize = 6
        self.State = 0
        self.previousAct = 0
        self.Price = 0
        self.Predictions = []
        self.OutputTxt = ""
        
    def loadData(self):
        column_name = ['open', 'high', 'low', 'close']
        training = pd.read_csv(self.TrainingFName, header=None, names=column_name)
        testing = pd.read_csv(self.TestingFName, header=None, names=column_name)
        return training, testing

    def preprocessing(self):
        trainingData = self.Training.copy();
        trainingData['mid'] = pd.DataFrame((trainingData['high'] + trainingData['low']) / 2)
        trainingData.dropna()
        # normailize
        for col in trainingData.columns:
            trainingData[col] = pd.DataFrame(self.Scaler.fit_transform(pd.DataFrame(trainingData[col])))
        return trainingWindows(trainingData, self.WindowSize)

    def definePredictData(self):
        total = pd.concat((self.Training, self.Testing), axis=0)
        total['mid'] = pd.DataFrame((total['high'] + total['low']) / 2)
        total.reset_index(inplace=True, drop=True)
        return total
    
    def predict(self, locat):
        predictData = []
        predictData.append(self.TPredictData[i - self.WindowSize:i])
        prediction = self.model.predict(predictData)
        return prediction[0], prediction[1]
    
    def makingAction(self, msg):
        return {
            'BUY': 'HOLD' if self.State == 1 else 'BUY',
            'HOLD': 'HOLD',
            'SELL': 'HOLD' if self.State == -1 else 'SELL',
        }[msg]

    def getAction(self, action):
        return {
            'BUY': 1,
            'HOLD': 0,
            'SELL':-1
        }[action]
    
    def decisionMaking(self, open, D1):
        action = 0
        if(self.previousAct != 0):
            self.Price = (float)(open[-1:])
            
        t = (float)(open[-1:])
            
        if(self.State == 0):
            if(D1 > t):
                action = self.getAction(self.makingAction('BUY'))
            elif(t > D1):
                action = self.getAction(self.makingAction('SELL'))
        else:
            if(self.State == 1):
                if(t < D1):
                    action = self.getAction(self.makingAction('HOLD'))
                elif(D1 < t):
                    action = self.getAction(self.makingAction('SELL'))
            elif(self.State == -1):
                if(self.Price < D1):
                    action = self.getAction(self.makingAction('HOLD'))
                elif(D1 < self.Price):
                    action = self.getAction(self.makingAction('BUY')) 
        
        self.State += action
        self.previousAct = action
        return action
    
    def main(self):
        # load data
        self.Training, self.Testing = self.loadData()
        # data pre-processing
        X_train, Y_train = self.preprocessing()
        
        # model building
        self.Model = modelBuilding(X_train.shape)
        callback = EarlyStopping(monitor="mean_absolute_error", patience=10, verbose=1, mode="auto")
        # model training
        self.Model.fit(X_train, Y_train, epochs=300, batch_size=32, validation_split=0.1, callbacks=[callback],shuffle=True)
        
        # define total predict data
        self.TPredictData = self.definePredictData()
        
        locat = len(self.Training)
        for i in range(locat + 1, locat + len(self.Testing)):
            data = self.TPredictData[0:i].copy()
            open = data['open'][-20:].copy()
            for col in data.columns:
                data[col] = pd.DataFrame(self.Scaler.fit_transform(pd.DataFrame(data[col])))
            predictData = [data[i - 6:i]]
            
            # predict
            prediction = self.Scaler.inverse_transform(self.Model.predict(np.array(predictData)))
            D1 = prediction[0][0]
            self.Predictions.append(D1)
            
            # decision making
            action = self.decisionMaking(open, D1)
            self.OutputTxt += f'{action}\n'


if __name__ == "__main__":
    #
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default="training_data.csv", help="input training data file name")
    parser.add_argument("--testing", default="testing_data.csv", help="input testing data file name")
    parser.add_argument("--output", default="output.csv", help="output file name")
    args = parser.parse_args()

    #
    trader = Trader(args)
    trader.main()
    with open(args.output, 'w') as f:
        f.writelines(trader.OutputTxt)
        f.close()
    # draw
    index = np.array([i for i in range(len(trader.Predictions))])
    plt.plot(index, np.array(trader.Testing['open'][:-1]), color='red', label='real_value')
    plt.plot(index, np.array(trader.Predictions), color='blue', label='predicted_value')
    plt.legend()
    plt.show()
