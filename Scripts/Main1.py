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
from keras.layers import LSTM, Bidirectional, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as MAE


# Path to saved model named: trained_model.h5
model_path = r'C:\Users\Admin\pracExc\Project_Bitcoin\trained_model.h5'

# Path to dataset named: Bitcoin_History.csv
data_path =  r"C:\Users\Admin\pracExc\Project_Bitcoin\Bitcoin_History.csv"

# Path to history file of a pretrained model named Trained.csv
path_to_history = r'C:\Users\Admin\pracExc\Project_Bitcoin\Trained.csv'



class Model: 
    """
    A class used to represent the neural network model

    ...

    Attributes
    ----------
     df : pandas.df
        the processed dataset as pd.df object 

    dataset, train_set, sc_train_set, test_set, sc_test_set : numpy 2d array
        2d array representing different parts od the dataset (sc stands for scaled).

    scaler : MinMaxScaler
        the type of scaler used to scale the data

    window_size : int
        the number of observations in each group (window).

    self.x_train, self.y_train : numpy ndarray
        input for training phase,  has the shape of (x , window_size, 5),
        output for training phase,  has the shape of (x, 5)

    self.x_val, self.y_val : numpy ndarray
        input for test phase,  has the shape of (x , window_size, 5),
        real output for test phase,  has the shape of (x, 5)

    """

    def __init__ (self):

        self.df, self.dataset, self.scaler, self.train_set,self.sc_train_set, self.test_set, self.sc_test_set = Model.preprocess(data_path)
        self.window_size = 8
        self.x_train, self.y_train = Model.split_sequence(self.sc_train_set, self.sc_train_set ,self.window_size)
        self.x_val, self.y_val = Model.split_sequence(self.sc_test_set , self.sc_test_set,  self.window_size)




    def predict_plot_future(self,times):

        """ predict number of future observations and plot a graph based on the results


        Parameters
        ----------
        times : int
            the number of observations to predict in the future
        
        """

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

        """ predict test observations and plot a graph based on the results along with the real observations
        
        """

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



    def retrieve_model (self,saved_model_path =r'C:\Users\Admin\pracExc\Project_Bitcoin\Saved_Model.h5' ):

        """ Load the saved model


        Parameters
        ----------
        saved_model_path : str
            the path to the saved model file named trained_model.h5 in local machine

        """
        
        saved_model = keras.models.load_model(saved_model_path)
        self.model = saved_model
        self.history = pd.read_csv(path_to_history)

    def train (self):

        """ train the model on preprocessed dataset

        """
        
        tf.keras.backend.clear_session()
        self.model = Model.make_model()
        self.history = self.model.fit(self.x_train, self.y_train, batch_size = 2048, epochs = 2)


    def plot_loss (self):

        """ plot the loss of training phase
        
        
        
        """
    
    # case 1 : the model was just trained

        try:
            his = self.history.history['loss']
            plt.figure(figsize=(8,6))
            plt.plot(his, 'o-', mfc='none', markersize=10, label='Train')
            plt.title('Trained Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

    # case 2: the model was already trained using data 

        except:
            
            
            his = self.history['loss']
            plt.figure(figsize=(8,6))
            plt.plot(his, 'o-', mfc='none', markersize=10, label='Train')
            plt.title('Prerained Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()


    def plot_mae (self):

        """ plot the MAE of training phase
        
        
        
        """
    
    # case 1 : the model was just trained

        try:
            his = self.history.history['mae']
            plt.figure(figsize=(8,6))
            plt.plot(his, 'r')
            plt.title('Trained Model MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            plt.show()

    # case 2: the model was already trained using data 

        except:
            
            
            his = self.history['mae']
            plt.figure(figsize=(8,6))
            plt.plot(his, 'r')
            plt.title('Prerained Model MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            plt.show()



    def plot_orgdata (self):

        """ plot the original data after preprocessing
        
        
        """

        self.df.plot( y='Weighted_Price')
        plt.title("Weighted price")
        plt.show()
    
    
    
    
    
    def print_shapes(self):
        """ print shapes of attributes

        """

        print("dataset.shape: {}".format(self.dataset.shape))
        print("df.shape: {}".format(self.df.shape))
        print("x_train.shape: {}".format(self.x_train.shape))
        print("y_train.shape: {}".format(self.y_train.shape))
        print("x_val.shape: {}".format(self.x_val.shape))
        print("y_val.shape: {}".format(self.y_val.shape))
        print('=======================')



    @staticmethod
    def preprocess (path_df=  r"C:\Users\Admin\pracExc\Project_Bitcoin\Bitcoin_History.csv", partition = 0.7):
        """ preprocess the dataset and create several attributes


        Parameters
        ----------
        path_df : str
            the path to the csv dataset named Bitcoin_History.csv in local machine
        
        partition : double
            represent the size of training data rational to the whole dataset (0 - 1)

        """
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

        dataset_size = dataset.shape[0]
        train_set = dataset[0: math.ceil(partition * dataset_size)]

        sc_train_set = scaler.fit_transform(train_set)

        test_set =  dataset[math.floor(partition * dataset_size):]

        sc_test_set = scaler.fit_transform(test_set)

        return df, dataset, scaler, train_set,sc_train_set, test_set, sc_test_set


    @staticmethod
    def split_sequence(sequence, to_predict, n_steps = 10):

        """ function that splits a dataset sequence into input data and output


        Parameters
        ----------
        sequence : numpy ndarray
             the input array of neural network to split

        to_predict : numpy ndarray
             the output array of neural network to split

        n_steps : int
            the window size 

        """
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

        """ return the neural network model object - Sequential 

        """

        tf.keras.backend.clear_session()
        # Define the model
        model = Sequential()
        model.add(Bidirectional(LSTM(units = 64, activation = 'tanh', return_sequences=True, input_shape=[None, 5] )))
        model.add(Bidirectional(LSTM(64, activation = 'tanh',  return_sequences=True)))
        model.add(Bidirectional(LSTM(64, activation= "tanh")))
        model.add(Dense(5))
        model.compile(loss= 'huber_loss', optimizer='adam',  metrics=["mae"])
    
        return model
    


