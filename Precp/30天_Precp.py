from math import sqrt
from numpy import concatenate
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
from keras.layers import Dropout
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
dataset = read_csv('Precp.csv', header=0, index_col=0)
testset = read_csv('2019-05.csv', header=0, index_col=0)
values = dataset.values
testvalues = testset.values
values = values.astype('float32')
testvalues = testvalues.astype('float32')
print(testvalues)
print('------------------------------------')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
testscaled = scaler.fit_transform(testvalues)
print(testscaled.shape)
# frame as supervised learning
reframed = series_to_supervised(scaled, 15, 1)
print(reframed.shape)

# drop columns we don't want to predict
i = 0
j=13
for i in range(13*15-1) :
	reframed.drop(reframed.columns[[j]], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values = reframed.values
print(values.shape)
print('------------------------------------')
n_train_hours = 2892+319
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
predict_test_X = testscaled
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]

print('------------------------------------')


test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
predict_test_X = predict_test_X.reshape((predict_test_X.shape[0], 1, predict_test_X.shape[1]))

print('------------------------------------')

# design network
model = Sequential()
model.add(LSTM(units  = 50, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
#model.add(Dropout(0.8))
model.add(LSTM(units = 50,return_sequences=True))
model.add(LSTM(units = 50,return_sequences=False))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=100,validation_data=(test_X, test_y), verbose=2, shuffle=False)

# make a prediction
yhat = model.predict(predict_test_X)
predict_test_X = predict_test_X.reshape((predict_test_X.shape[0], predict_test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((predict_test_X[:, 1:],yhat), axis=1)
inv_yhat = inv_yhat[:,-13:]
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,12]
print(inv_yhat)

datetime_object = datetime.strptime('2019/06/01','%Y/%m/%d')
with open('pred_Precp.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(['date', 'pred_Precp'])
	k = 1
	for i in inv_yhat :
		writer.writerow([datetime_object,i])
		datetime_object = datetime_object + timedelta(days=1)