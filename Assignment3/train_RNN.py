# import required packages
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

if __name__ == "__main__": 
	"""#read dataset
	data = pd.read_csv('Assignment3/assignment 3/data/q2_dataset.csv')
	data.columns = data.columns.str.strip()

	#create targets from Opening price
	data['target']= data['Open']
	#convert dates to date datatype
	data['Date'] =pd.to_datetime(data.Date)
	data=data.sort_values(by='Date')
	#feature creation 
	data['Volume_t-3'] = data.shift(3)['Volume']
	data['Volume_t-2'] = data.shift(2)['Volume']
	data['Volume_t-1'] = data.shift(1)['Volume']
	data['Open_t-3'] = data.shift(3)['Open']
	data['Open_t-2'] = data.shift(2)['Open']
	data['Open_t-1'] = data.shift(1)['Open']
	data['High_t-3'] = data.shift(3)['High']
	data['High_t-2'] = data.shift(2)['High']
	data['High_t-1'] = data.shift(1)['High']
	data['Low_t-3'] = data.shift(3)['Low']
	data['Low_t-2'] = data.shift(2)['Low']
	data['Low_t-1'] = data.shift(1)['Low']
	data['target']= data['Open']

	#remove redundant features and null values
	data = data.drop(['Close/Last','Volume','Open','High','Low'], axis = 1)
	data = data.dropna()

	data = data[[
	'Date',
	'Volume_t-3',
	'Volume_t-2',
	'Volume_t-1',
	'Open_t-3',
	'Open_t-2',
	'Open_t-1',
	'High_t-3',
	'High_t-2',
	'High_t-1',
	'Low_t-3',
	'Low_t-2',
	'Low_t-1', 
	'target']]

	#split the data into train and test set
	train, test = train_test_split(data, test_size=0.30, random_state=0)
	#save the data
	train.to_csv('Assignment3/assignment 3/data/train_data_RNN.csv',index=False)
	test.to_csv('Assignment3/assignment 3/data/test_data_RNN.csv',index=False)"""
	#end comments

	# 1. load your training data
	data_train = pd.read_csv('Assignment3/assignment 3/data/train_data_RNN.csv')
	data_test = pd.read_csv('Assignment3/assignment 3/data/test_data_RNN.csv')

	#separate features and target
	X_train = data_train.drop(['Date','target'], axis = 1)
	y_train = data_train['target']


	#preprocessing
	scaler=MinMaxScaler(feature_range=(0,1))
	X_train=scaler.fit_transform(X_train)

	#saving the scaler to apply it on the test dataset
	with open('scaler_RNN_model','wb') as file_pick:
		pickle.dump(scaler,file_pick)

	#numpy array conversion
	X_train=np.array(X_train)

	# reshape input to be [samples, time steps, features] which is required for LSTM
	X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)

	#model construction
	model = tf.keras.models.Sequential([
	# Shape [batch, time, features] => [batch, time, lstm_units]
	tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(12,1)),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.LSTM(50, return_sequences=True),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.LSTM(50),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(units=1)])

	model.compile(loss='mean_squared_error',optimizer='adam')
	monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto',
		restore_best_weights=True)
	# 2. Train your network
	# 		Make sure to print your training loss within training to show progress
	# 		Make sure you print the final training loss
	history = model.fit(X_train,y_train,validation_split=0.05,epochs=1500,batch_size=64,verbose=1)
	#print final training loss here
	print('Final Training Loss: \t', history.history["loss"][-1] )
	# 3. Save your model
	model.save('Assignment3/assignment 3/models/Group3_RNN_model.h5')