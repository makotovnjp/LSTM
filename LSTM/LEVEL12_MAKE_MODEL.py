
# coding: utf-8

# In[1]:


#import は不要なモノも入っている可能性大です

import pandas as pd
import numpy as np
from datetime import datetime as dt
import math

from pandas.core import resample

from keras.optimizers import SGD

import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.layers import LSTM

from keras.callbacks import EarlyStopping, CSVLogger

from pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')

#from sklearn.model_selection import train_test_split


# In[2]:


input_output_columns = 5  # 取り込んだデータの左から何項目を予測値（学習時はＹ）とするか。
look_back = 8 


# In[3]:


import tensorflow as tf
tf.reset_default_graph()


# In[4]:


# 為替予測

#filename = '~/Documents/ML/tensorflow/csv/GBPJPY_M15.csv'
filename = '~/Documents/tensorflow/csv/LEVEL12.csv'

# 
df = pd.read_csv(filename,
                nrows=100000,
                usecols=['Date','Week',
                        'Open','High','Low','Close','Volume','Profit','HighLowDiff','OpenRelativePosition','CloseRelativePosition',
                        'RCIs','RCIm','RCIl','ADXdm','ADXdp','ADXadx','MACD','MACDsig'],
                dtype={ 'Date': str,
                        'Week':float,
                        'Open':float,'High':float,'Low':float,'Close':float,'Volume':float,'Profit':float,
                        'HighLowDiff':float,'OpenRelativePosition':float,'CloseRelativePosition':float,
                        'RCIs':float,'RCIm':float,'RCIl':float,'ADXdm':float,'ADXdp':float,'ADXadx':float,'MACD':float,'MACDsig':float},
                parse_dates=[0]
            ).sort_values(by='Date', ascending=True)


# In[5]:


# 日付と時間のデータが分かれているので、一つにつなげて日時型に変換し、インデックスに指定
#df['Datetime'] = pd.to_datetime(df['Date']+' '+df['Time'])

df.index = df['Date']
df = df[df.index.year>=2010]
df.sort_index(ascending=True)

df.dropna(inplace=True) #2つ以上のNanは削除。inplaceは自身を書き換え


# In[6]:


#df['Week']=df['Week'].astype('float32')
print(df.info())


# In[7]:


print(df[:50])


# In[8]:


print(df.info())

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        xset = []
        for j in range(dataset.shape[1]):
            a = dataset[i:(i+look_back), j]
            xset.append(a)
        dataY.append(dataset[i + look_back, ])      
        dataX.append(xset)
    return numpy.array(dataX), numpy.array(dataY)


# In[9]:


# Yは最初の列に配置
#dataframe = df.loc[:,['Close','High','Low','Volume','HighLowDiff','OpenRelativePosition','CloseRelativePosition','Week','Hourminutes']].dropna()

dataframe = df.loc[:,[ 'Open','High','Low','Close','ADXadx','HighLowDiff','OpenRelativePosition','CloseRelativePosition']].dropna()


#dataframe = df.loc[:,['ExpectedPrice','Close',]].dropna()

print(dataframe[:5])   #  一番左にDateが出るがIndexであるため無視して良い


# In[10]:


dataset = dataframe.values
dataset = dataset.astype('float64')
#dataset = dataset.astype('str')

from sklearn.preprocessing import MinMaxScaler

# データを標準化します。LSTMはセンシティブなので扱う数値を標準化されなければならない
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

print(scaler)

# データのうちの0.2をテスト用
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# In[11]:


trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# In[12]:


print(trainX.shape)
print(trainY.shape)

print(testX.shape)
print(testY.shape)

#print(trainX[:6]) 
print('------')
print(trainY[:6]) 

print(trainY.shape[1])

trainY_tmp=np.delete(trainY,range(input_output_columns,trainY.shape[1]),1)
testY_tmp=np.delete(testY,range(input_output_columns,testY.shape[1]),1)

print(trainX.shape)
print(trainY.shape)

trainY=trainY_tmp
testY=testY_tmp


# In[13]:


early_stopping = EarlyStopping(patience=30, verbose=1)


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(100, input_shape=(testX.shape[1], look_back)))
    
model.add(Dense(input_output_columns))

model.add(Activation("hard_sigmoid"))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
model_history=model.fit(trainX, trainY, epochs=500, batch_size=32, verbose=2, validation_data=(testX,testY),callbacks=[early_stopping])
#model_history=model.fit(trainX, trainY, epochs=500, batch_size=32, verbose=2, validation_data=(testX,testY))

print(model.summary())


# In[14]:


# モデルのsave/load

import pickle
model.save('level12_model.h5')
with open('level12_model.pkl', 'wb') as f:
    pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    
score = model.evaluate(testX, testY, verbose=0)
print("Test score:" + str(score[0]))
print("Test accuracy:" + str(score[1]))


# In[15]:



from sklearn.metrics import mean_squared_error


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[16]:


print(trainPredict.shape)
print(testPredict.shape)


# In[17]:


print('trainPredict.shape='+str(trainPredict.shape))
print('testPredict.shape='+str(testPredict.shape))
print('trainX.shape='+str(trainX.shape))
print('trainY.shape='+str(trainY.shape))
print('------')
print(trainX.shape[1])
print('------')


train_zero=np.resize(trainX,(trainX.shape[0],trainX.shape[1]))[:,input_output_columns:]  # scaler実行時のrow/column-1 をつくり、そこからinput_output_columnsの右の列を追加
train_zero.fill(0)
trainPredict = np.hstack([trainPredict,train_zero])
trainPredict = scaler.inverse_transform(trainPredict)

trainY = np.hstack([trainY,train_zero])
trainY = scaler.inverse_transform(trainY)


test_zero=np.resize(testX,(testY.shape[0],testX.shape[1]))[:,input_output_columns:]  # scaler実行時のrow/column-1 をつくり、そこからinput_output_columnsの右の列を追加
test_zero.fill(0)

testPredict = np.hstack([testPredict,test_zero])
testPredict = scaler.inverse_transform(testPredict)

testY = np.hstack([testY,test_zero])
testY = scaler.inverse_transform(testY)

print(trainPredict.shape)
print(testPredict.shape)


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

acc = model_history.history['acc']
loss = model_history.history['loss']


# In[18]:


print(acc[-1])


# In[19]:


acc = model_history.history['acc']
loss = model_history.history['loss']

print(acc[-1])
print(loss[-1])


# In[20]:


#print(testY[:,0])
#print(testPredict[:,0])

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(dataset)-len(testPredict):len(dataset), :] = testPredict


# plot baseline and predictions
# predict 結果の表示。概ね良い感じ。完全な一致は求めていないのでＯＫとする。
rcParams['figure.figsize'] = 20,10
#plt.plot(scaler.inverse_transform(dataset))
tra=scaler.inverse_transform(dataset)
plt.plot(tra[:,0])
plt.plot(trainPredictPlot[:,0])
plt.plot(testPredictPlot[:,0])
plt.show()


# In[21]:


fig, ax1 = plt.subplots(1,1)
 
ax1.plot(model_history.epoch, model_history.history['loss'])
ax1.set_title('TrainingError')
 
if model.loss == 'mae':
    ax1.set_ylabel('Mean Absolute Error (MAE)',fontsize=12)
# もでるおんロス計算を変更した場合のため
else:
    ax1.set_ylabel('Model Loss',fontsize=12)
ax1.set_xlabel('# Epochs',fontsize=12)
plt.show()
 


# In[22]:


# 損失の履歴をプロット
plt.clf()
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.legend(['loss', 'val_loss'], loc='lower right')
plt.show()

# チャート作成(ACC/LOSS)
plt.clf()
epochs = len(loss)
plt.plot(range(epochs), acc, marker='.', label='acc')
plt.plot(range(epochs), loss, marker='.', label='loss')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()


# #### 
