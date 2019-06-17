from math import sqrt
from numpy import concatenate
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import csv
from datetime import datetime
from datetime import timedelta
 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
dataset = read_csv('Temperature.csv', header=0, index_col=0)
testset = read_csv('2019-05.csv', header=0, index_col=0)
values = dataset.values
testvalues = testset.values
# ensure all data is float
values = values.astype('float32')
testvalues = testvalues.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
testscaled = scaler.fit_transform(testvalues)
# frame as supervised learning
reframed = series_to_supervised(scaled, 30, 1)
# drop columns we don't want to predict
i = 0
j=3
for i in range(3*29) :
	reframed.drop(reframed.columns[[j]], axis=1, inplace=True)
print(reframed.head())
print(reframed.shape)
# split into train and test sets
values = reframed.values
n_train_hours = 3344
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-3], train[:, -3:]
test_X, test_y = test[:, :-3], test[:, -3:]
predtest_X = testscaled
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
predtest_X = predtest_X.reshape((predtest_X.shape[0], 1, predtest_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# design network
model = Sequential()
model.add(LSTM(units  = 100, input_shape=(train_X.shape[1],train_X.shape[2]),activation='relu',return_sequences=True))
model.add(LSTM(units  = 100,return_sequences=True))
model.add(LSTM(units = 100,activation='relu',return_sequences=False))
model.add(Dense(3))
model.compile(loss='mae', optimizer='RMSprop')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=30, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
# make a prediction
#yhat = model.predict(test_X)
#test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
#inv_yhat = concatenate((test_X[:, 1:],yhat ), axis=1)
yhat = model.predict(predtest_X)
predtest_X = predtest_X.reshape((predtest_X.shape[0], predtest_X.shape[2]))

# invert scaling for forecast
inv_yhat = scaler.inverse_transform(yhat)
inv_yhat = np.round(inv_yhat,1)
temperature = inv_yhat[:,-1]
RH = inv_yhat[:,-2]
WS = inv_yhat[:,-3]
print(temperature)
print(RH)
print(WS)

datetime_object = datetime.strptime('2019/06/01','%Y/%m/%d')
with open('pred_weather.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(['date', 'pred_temperature','pred_RH','pred_WS'])
	k = 1
	for i in inv_yhat :
		writer.writerow([datetime_object,i[-1],i[-2],i[-3]])
		datetime_object = datetime_object + timedelta(days=1)