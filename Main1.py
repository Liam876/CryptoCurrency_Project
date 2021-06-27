from keras.engine.saving import load_model
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import math
import keras
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense, Conv1D, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as MAE


model_path = r'C:\Users\Admin\pracExc\Project_Bitcoin\trained_model.h5'
data_path =  r"C:\Users\Admin\pracExc\Project_Bitcoin\Bitcoin_History.csv"

class Model: 

    def __init__ (self):
        self.model = Model.make_model()
        self.df, self.dataset, self.scaler, self.train_set,self.sc_train_set, self.test_set, self.sc_test_set = Model.preprocess(data_path)
        self.window_size = 8
        self.x_train, self.y_train = Model.split_sequence(self.sc_train_set, self.sc_train_set ,self.window_size)
        self.x_val, self.y_val = Model.split_sequence(self.sc_test_set , self.sc_test_set,  self.window_size)




    def predict_plot_future(self,times):
        fut = self.x_val[-1:]  # Take last window
        Y = []
        for i in range (0, times):
            cur_window = fut[i]
            #print(cur_window)
            y = self.model.predict(np.expand_dims(cur_window, axis = 0))
            #print(y[0])
            Y.append(y[0])
            #print(y)
            new_window = np.append(cur_window[ 1: ,] , y, axis = 0)
            #print(new_window.shape)
            fut = np.append(fut, np.expand_dims(new_window, axis = 0), axis = 0)
            #print(fut.shape)

       
        #print(np.array(Y).shape)
        #print(np.array(Y)[: , -1])
        self.scaler.fit(self.test_set[: , -1].reshape(-1,1))
        new_fut_y = self.scaler.inverse_transform(np.array(Y)[: ,-1].reshape(-1,1)).reshape(1,-1)
        #print(new_fut_y, new_fut_y.shape)
        print(new_fut_y[0])
        #plt.plot(np.array(Y)[: , -1])
        plt.plot(new_fut_y[0], color = 'green',marker = 'o',  markerfacecolor='blue', markersize=9)
        plt.show()


    def predict_plot_test(self):
        sc_preds = self.model.predict(self.x_val)
        #print(preds[-10:])
        y_org =  self.test_set[ : , -1]
        self.scaler.fit(y_org.reshape(-1,1))
        preds = self.scaler.inverse_transform(sc_preds[: , -1].reshape(-1,1))

        #print(preds[-10:])

        plt.plot(preds,label = "predicted", color = 'orange')
        plt.plot(y_org[self.window_size:], label = "real", color = 'blue')
        plt.title("Test phase - predicting the test results")
        plt.legend()
        plt.show()



    def retrieve_model (self,saved_model_path =r'C:\Users\Admin\pracExc\Project_Bitcoin\trained_model.h5' ):
        saved_model = keras.models.load_model(saved_model_path)
        self.model = saved_model

    def train (self):
        # Train model

        #tf.keras.backend.clear_session()

        self.history = self.model.fit(self.x_train, self.y_train, batch_size = 2048, epochs = 100)



    def print_shapes(self):
        # Print Shapes

        print("dataset.shape: {}".format(self.dataset.shape))
        print("df.shape: {}".format(self.df.shape))
        print("x_train.shape: {}".format(self.x_train.shape))
        print("y_train.shape: {}".format(self.y_train.shape))
        print("x_val.shape: {}".format(self.x_val.shape))
        print("y_val.shape: {}".format(self.y_val.shape))
        print('=======================')



    @staticmethod
    def preprocess (path_df=  r"C:\Users\Admin\pracExc\Project_Bitcoin\Bitcoin_History.csv", partition = 0.7):
        df = pd.read_csv(path_df)

        df['Timestamp'] = pd.to_datetime(df['Timestamp'],infer_datetime_format=True, unit='s')

        df = df.set_index('Timestamp')

        df = df.drop([ 'Volume_(BTC)', 'Open' ], axis=1)

        df = df.reindex(columns=['High','Low', 'Close', 'Volume_(Currency)', 'Weighted_Price'])

        df['Close'] = df['Close'].resample('1H').last()
        df['High'] = df['High'].resample('1H').last()
        df['Low'] = df['Low'].resample('1H').last()
        df['Volume_(Currency)'] = df['Volume_(Currency)'].resample('1H').sum()
        df['Weighted_Price'] = df['Weighted_Price'].resample('1H').last()


        df = df.dropna()
        dataset = df.values

        scaler = MinMaxScaler(feature_range=(0, 1))
        #dataset = scaler.fit_transform(dataset)

        dataset_size = dataset.shape[0]
        train_set = dataset[0: math.ceil(partition * dataset_size)]

        sc_train_set = scaler.fit_transform(train_set)

        test_set =  dataset[math.floor(partition * dataset_size):]

        sc_test_set = scaler.fit_transform(test_set)

        return df, dataset, scaler, train_set,sc_train_set, test_set, sc_test_set


    @staticmethod
    def split_sequence(sequence, to_predict, n_steps = 10):
        # function that splits a dataset sequence into input data and labels
        X, Y = [], []
        for i in range(sequence.shape[0]):
            if (i + n_steps) >= sequence.shape[0]:
                break
            # Divide sequence between data (input) and labels (output)
            seq_X, seq_Y = sequence[i: i + n_steps],to_predict[i + n_steps]
            X.append(seq_X)
            Y.append(seq_Y)
        return np.array(X), np.array(Y)


    @staticmethod
    def make_model ():

        tf.keras.backend.clear_session()
        # Define the model
        model = Sequential()
        #model.add(Conv1D(filters=32, kernel_size=10, strides=1, padding="causal", activation="tanh", input_shape=[None, 5]))
        model.add(Bidirectional(LSTM(units = 64, activation = 'tanh', return_sequences=True, input_shape=[None, 5] )))
        model.add(Bidirectional(LSTM(64, activation = 'tanh',  return_sequences=True)))
        model.add(Bidirectional(LSTM(64, activation= "tanh")))
        #model.add(Dense(30, activation= "tanh"))
        #model.add(Dense(10, activation= "tanh"))
        model.add(Dense(5))
        model.compile(loss= 'huber_loss', optimizer='adam',  metrics=["mae"])
    
        return model
    



#test = Model()
#from keras.utils.vis_utils import plot_model

#plot_model(test.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#test.print_shapes()
#test.train()
#test.model.save(r'C:\Users\Admin\pracExc\Project_Bitcoin\trained_model.h5')
#saved_m = keras.models.load_model(r'C:\Users\Admin\pracExc\Project_Bitcoin\trained_model.h5')
#saved_m.summary()
#print(test.model.predict(test.x_val[-1:]))
#test.retrieve_model()
#test.predict_plot_future(1)


#print(test.x_val[-1:][0].shape)