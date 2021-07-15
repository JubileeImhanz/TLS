# import required packages
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__":
	# 1. Load your saved model
	model = tf.keras.models.load_model('Assignment3/assignment 3/models/Group3_RNN_model.h5')

	# 2. Load your testing data
	data_test = pd.read_csv('test_data_RNN.csv')

	#separate features and target
	X_test_date = data_test
	X_test = data_test.drop(['Date','target'], axis = 1)
	y_test = data_test['target']

	#preprocessing
	scaler=MinMaxScaler(feature_range=(0,1))
	X_test=scaler.fit_transform(X_test)

	#numpy array conversion
	X_test=np.array(X_test)
	y_test=np.array(y_test)
	# reshape input to be [samples, time steps, features] which is required for LSTM
	X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
	# 3. Run prediction on the test data and output required plot and loss
	y_pred = model.predict(X_test)

	#print Test Loss
	print('Test Loss:\t ', mean_squared_error(y_pred, y_test))
	#create dataframe for plotting against date
	result_array=pd.DataFrame({'y_test':y_test, 'y_predicted':y_pred.ravel(),'Date':X_test_date["Date"]},index=None)
	#result_array=result_array.reset_index(drop=True, inplace=False)
	result_array['Date'] =pd.to_datetime(result_array.Date)
	result_array=result_array.sort_values(by='Date')
	result_array=result_array.reset_index(drop=True, inplace=False)

	#plot data
	result_array.iloc[0:,0:2].plot.line(figsize=(13,8))
	plt.xticks(np.arange(0, 377, step=20), result_array["Date"].dt.date.iloc[lambda x: x.index % 20 == 0],rotation=45)
	plt.xlabel('Date')
	plt.ylabel('Opening price')
	plt.title('Actual vs Predicted')
	plt.show()