from flask import Flask, render_template
from config import DevConfig
from flask import json
import numpy as np
#import matplotlib.pyplot as plt
import csv
import math
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
#=======================================================================這裡做Precp的計算，除了少部分可共用的參數外，大部分參數都在後面加上Pre做區分
datasetPre = read_csv('Precp/Precp.csv', header=0, index_col=0)
testsetPre = read_csv('Precp/2019-05.csv', header=0, index_col=0)
valuesPre = datasetPre.values
testvaluesPre = testsetPre.values
valuesPre = valuesPre.astype('float32')
testvaluesPre = testvaluesPre.astype('float32')
print(testvaluesPre)
print('------------------------------------')
# normalize features
scalerPre = MinMaxScaler(feature_range=(0, 1))
scaledPre = scalerPre.fit_transform(valuesPre)
testscaledPre = scalerPre.fit_transform(testvaluesPre)
print(testscaledPre.shape)
# frame as supervised learning
reframedPre = series_to_supervised(scaledPre, 15, 1)
print(reframedPre.shape)

# drop columns we don't want to predict
i = 0
j=13
for i in range(13*15-1) :
	reframedPre.drop(reframedPre.columns[[j]], axis=1, inplace=True)
print(reframedPre.head())

# split into train and test sets
valuesPre = reframedPre.values
print(valuesPre.shape)
print('------------------------------------')
n_train_hours = 2892+319
trainPre = valuesPre[:n_train_hours, :]
testPre = valuesPre[n_train_hours:, :]
predict_test_XPre = testscaledPre
# split into input and outputs
train_XPre, train_yPre = trainPre[:, :-1], trainPre[:, -1]

print('------------------------------------')


test_XPre, test_yPre = testPre[:, :-1], testPre[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_XPre = train_XPre.reshape((train_XPre.shape[0], 1, train_XPre.shape[1]))
test_XPre = test_XPre.reshape((test_XPre.shape[0], 1, test_XPre.shape[1]))
predict_test_XPre = predict_test_XPre.reshape((predict_test_XPre.shape[0], 1, predict_test_XPre.shape[1]))

print('------------------------------------')

# design network
modelPre = Sequential()
modelPre.add(LSTM(units  = 50, input_shape=(train_XPre.shape[1], train_XPre.shape[2]),return_sequences=True))
#model.add(Dropout(0.8))
modelPre.add(LSTM(units = 50,return_sequences=True))
modelPre.add(LSTM(units = 50,return_sequences=False))
modelPre.add(Dense(1))
modelPre.compile(loss='mae', optimizer='adam')
# fit network
historyPre = modelPre.fit(train_XPre, train_yPre, epochs=50, batch_size=100,validation_data=(test_XPre, test_yPre), verbose=2, shuffle=False)

# make a prediction
yhatPre = modelPre.predict(predict_test_XPre)
predict_test_XPre = predict_test_XPre.reshape((predict_test_XPre.shape[0], predict_test_XPre.shape[2]))
# invert scaling for forecast
inv_yhatPre = concatenate((predict_test_XPre[:, 1:],yhatPre), axis=1)
inv_yhatPre = inv_yhatPre[:,-13:]
inv_yhatPre = scalerPre.inverse_transform(inv_yhatPre)
inv_yhatPre = inv_yhatPre[:,12]
print(inv_yhatPre)

datetime_object = datetime.strptime('2019/06/01','%Y/%m/%d')
with open('Precp/pred_Precp.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(['date', 'pred_Precp'])
	k = 1
	for i in inv_yhatPre :
		writer.writerow([datetime_object,i])
		datetime_object = datetime_object + timedelta(days=1)

#=========================================================================這裡做RH的計算，除了少部分可共用的參數外，大部分參數都在後面加上RH做區分

datasetRH = read_csv('RH/RH.csv', header=0, index_col=0)
testsetRH = read_csv('RH/2019-05.csv', header=0, index_col=0)
valuesRH = datasetRH.values
testvaluesRH = testsetRH.values
valuesRH = valuesRH.astype('float32')
testvaluesRH = testvaluesRH.astype('float32')
print(testvaluesRH)
print('------------------------------------')
# normalize features
scalerRH = MinMaxScaler(feature_range=(0, 1))
scaledRH = scalerRH.fit_transform(valuesRH)
testscaledRH = scalerRH.fit_transform(testvaluesRH)
print(testscaledRH.shape)
# frame as supervised learning
reframedRH = series_to_supervised(scaledRH, 15, 1)
print(reframedRH.shape)

# drop columns we don't want to predict
i = 0
j=13
for i in range(13*15-1) :
	reframedRH.drop(reframedRH.columns[[j]], axis=1, inplace=True)
print(reframedRH.head())

# split into train and test sets
valuesRH = reframedRH.values
print(valuesRH.shape)
print('------------------------------------')
n_train_hours = 2892+319
trainRH = valuesRH[:n_train_hours, :]
testRH = valuesRH[n_train_hours:, :]
predict_test_XRH = testscaledRH
# split into input and outputs
train_XRH, train_yRH = trainRH[:, :-1], trainRH[:, -1]

print('------------------------------------')


test_XRH, test_yRH = testRH[:, :-1], testRH[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_XRH = train_XRH.reshape((train_XRH.shape[0], 1, train_XRH.shape[1]))
test_XRH = test_XRH.reshape((test_XRH.shape[0], 1, test_XRH.shape[1]))
predict_test_XRH = predict_test_XRH.reshape((predict_test_XRH.shape[0], 1, predict_test_XRH.shape[1]))

print('------------------------------------')

# design network
modelRH = Sequential()
modelRH.add(LSTM(units  = 50, input_shape=(train_XRH.shape[1], train_XRH.shape[2]),return_sequences=True))
#modelRH.add(Dropout(0.8))
modelRH.add(LSTM(units = 50,return_sequences=True))
modelRH.add(LSTM(units = 50,return_sequences=False))
modelRH.add(Dense(1))
modelRH.compile(loss='mae', optimizer='adam')
# fit network
historyRH = modelRH.fit(train_XRH, train_yRH, epochs=50, batch_size=100,validation_data=(test_XRH, test_yRH), verbose=2, shuffle=False)

# make a prediction
yhatRH = modelRH.predict(predict_test_XRH)
predict_test_XRH = predict_test_XRH.reshape((predict_test_XRH.shape[0], predict_test_XRH.shape[2]))
# invert scaling for forecast
inv_yhatRH = concatenate((predict_test_XRH[:, 1:],yhatRH), axis=1)
inv_yhatRH = inv_yhatRH[:,-13:]
inv_yhatRH = scalerRH.inverse_transform(inv_yhatRH)
inv_yhatRH = inv_yhatRH[:,12]
print(inv_yhatRH)

datetime_object = datetime.strptime('2019/06/01','%Y/%m/%d')
with open('RH/pred_RH.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(['date', 'pred_RH'])
	k = 1
	for i in inv_yhatRH :
		writer.writerow([datetime_object,i])
		datetime_object = datetime_object + timedelta(days=1)
#================================================================這裡做TM的計算，除了少部分可共用的參數外，大部分參數都在後面加上TM做區分
datasetTM = read_csv('Temperature/Temperature.csv', header=0, index_col=0)
testsetTM = read_csv('Temperature/2019-05.csv', header=0, index_col=0)
valuesTM = datasetTM.values
testvaluesTM = testsetTM.values
valuesTM = valuesTM.astype('float32')
testvaluesTM = testvaluesTM.astype('float32')
print(testvaluesTM)
print('------------------------------------')
# normalize features
scalerTM = MinMaxScaler(feature_range=(0, 1))
scaledTM = scalerTM.fit_transform(valuesTM)
testscaledTM = scalerTM.fit_transform(testvaluesTM)
print(testscaledTM.shape)
# frame as supervised learning
reframedTM = series_to_supervised(scaledTM, 15, 1)
print(reframedTM.shape)

# drop columns we don't want to predict
i = 0
j=13
for i in range(13*15-1) :
	reframedTM.drop(reframedTM.columns[[j]], axis=1, inplace=True)
print(reframedTM.head())

# split into train and test sets
valuesTM = reframedTM.values
print(valuesTM.shape)
print('------------------------------------')
n_train_hours = 2892+319
trainTM = valuesTM[:n_train_hours, :]
testTM = valuesTM[n_train_hours:, :]
predict_test_XTM = testscaledTM
# split into input and outputs
train_XTM, train_yTM = trainTM[:, :-1], trainTM[:, -1]

print('------------------------------------')


test_XTM, test_yTM = testTM[:, :-1], testTM[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_XTM = train_XTM.reshape((train_XTM.shape[0], 1, train_XTM.shape[1]))
test_XTM = test_XTM.reshape((test_XTM.shape[0], 1, test_XTM.shape[1]))
predict_test_XTM = predict_test_XTM.reshape((predict_test_XTM.shape[0], 1, predict_test_XTM.shape[1]))

print('------------------------------------')

# design network
modelTM = Sequential()
modelTM.add(LSTM(units  = 50, input_shape=(train_XTM.shape[1], train_XTM.shape[2]),return_sequences=True))
#modelTM.add(Dropout(0.8))
modelTM.add(LSTM(units = 50,return_sequences=True))
modelTM.add(LSTM(units = 50,return_sequences=False))
modelTM.add(Dense(1))
modelTM.compile(loss='mae', optimizer='adam')
# fit network
historyTM = modelTM.fit(train_XTM, train_yTM, epochs=50, batch_size=100,validation_data=(test_XTM, test_yTM), verbose=2, shuffle=False)

# make a prediction
yhatTM = modelTM.predict(predict_test_XTM)
predict_test_XTM = predict_test_XTM.reshape((predict_test_XTM.shape[0], predict_test_XTM.shape[2]))
# invert scaling for forecast
inv_yhatTM = concatenate((predict_test_XTM[:, 1:],yhatTM), axis=1)
inv_yhatTM = inv_yhatTM[:,-13:]
inv_yhatTM = scalerTM.inverse_transform(inv_yhatTM)
inv_yhatTM = inv_yhatTM[:,12]
print(inv_yhatTM)

datetime_object = datetime.strptime('2019/06/01','%Y/%m/%d')
with open('Temperature/pred_temperature.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(['date', 'pred_temperature'])
	k = 1
	for i in inv_yhatTM :
		writer.writerow([datetime_object,i])
		datetime_object = datetime_object + timedelta(days=1)
#======================================================================這裡做WS的計算，除了少部分可共用的參數外，大部分參數都在後面加上WS做區分
datasetWS = read_csv('WS/WS.csv', header=0, index_col=0)
testsetWs = read_csv('WS/2019-05.csv', header=0, index_col=0)
valuesWS = datasetWS.values
testvaluesWS = testsetWs.values
valuesWS = valuesWS.astype('float32')
testvaluesWS = testvaluesWS.astype('float32')
print(testvaluesWS)
print('------------------------------------')
# normalize features
scalerWS = MinMaxScaler(feature_range=(0, 1))
scaledWS = scalerWS.fit_transform(valuesWS)
testscaledWS = scalerWS.fit_transform(testvaluesWS)
print(testscaledWS.shape)
# frame as supervised learning
reframedWS = series_to_supervised(scaledWS, 15, 1)
print(reframedWS.shape)

# drop columns we don't want to predict
i = 0
j=13
for i in range(13*15-1) :
	reframedWS.drop(reframedWS.columns[[j]], axis=1, inplace=True)
print(reframedWS.head())

# split into train and test sets
valuesWS = reframedWS.values
print(valuesWS.shape)
print('------------------------------------')
n_train_hours = 2892+319
trainWS = valuesWS[:n_train_hours, :]
testWS = valuesWS[n_train_hours:, :]
predict_test_XWS = testscaledWS
# split into input and outputs
train_XWS, train_yWS = trainWS[:, :-1], trainWS[:, -1]

print('------------------------------------')


test_XWS, test_yWS = testWS[:, :-1], testWS[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_XWS = train_XWS.reshape((train_XWS.shape[0], 1, train_XWS.shape[1]))
test_XWS = test_XWS.reshape((test_XWS.shape[0], 1, test_XWS.shape[1]))
predict_test_XWS = predict_test_XWS.reshape((predict_test_XWS.shape[0], 1, predict_test_XWS.shape[1]))

print('------------------------------------')

# design network
modelWS = Sequential()
modelWS.add(LSTM(units  = 50, input_shape=(train_XWS.shape[1], train_XWS.shape[2]),return_sequences=True))
#modelWS.add(Dropout(0.8))
modelWS.add(LSTM(units = 50,return_sequences=True))
modelWS.add(LSTM(units = 50,return_sequences=False))
modelWS.add(Dense(1))
modelWS.compile(loss='mae', optimizer='adam')
# fit network
historyWS = modelWS.fit(train_XWS, train_yWS, epochs=50, batch_size=100,validation_data=(test_XWS, test_yWS), verbose=2, shuffle=False)

# make a prediction
yhatWS = modelWS.predict(predict_test_XWS)
predict_test_XWS = predict_test_XWS.reshape((predict_test_XWS.shape[0], predict_test_XWS.shape[2]))
# invert scaling for forecast
inv_yhatWS = concatenate((predict_test_XWS[:, 1:],yhatWS), axis=1)
inv_yhatWS = inv_yhatWS[:,-13:]
inv_yhatWS = scalerWS.inverse_transform(inv_yhatWS)
inv_yhatWS = inv_yhatWS[:,12]
print(inv_yhatWS)

datetime_object = datetime.strptime('2019/06/01','%Y/%m/%d')
with open('WS/pred_WS.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(['date', 'pred_WS'])
	k = 1
	for i in inv_yhatWS :
		writer.writerow([datetime_object,i])
		datetime_object = datetime_object + timedelta(days=1)
# 路由和處理函式配對
app = Flask(__name__)
app.config.from_object(DevConfig)
@app.route('/')
def index():	#在這裡傳送資料
	dataPre = inv_yhatPre.tolist();
	dataRH = inv_yhatRH.tolist();
	dataTM = inv_yhatTM.tolist();
	dataWS = inv_yhatWS.tolist();
	#data = {'inv_yhatPre':inv_yhatPre[0],'inv_yhatRH':inv_yhatRH[0],'inv_yhatTM':inv_yhatTM[0],'inv_yhatWS':inv_yhatWS[0]};
	return render_template("front.html",dataPre=json.dumps(dataPre),dataRH=json.dumps(dataRH),dataTM=json.dumps(dataTM),dataWS=json.dumps(dataWS));

# 判斷自己執行非被當做引入的模組，因為 __name__ 這變數若被當做模組引入使用就不會是 __main__
if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False);