# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 16:54:47 2023

@author: Oumaima
"""



from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, GRU
from keras.layers import Dropout



# selecting 100 different users
# selecting 50 different web services
# merging invocations with information about users and web 


# selecting 10 different users

userdatapd = pd.read_csv('C:/Users/Oumaima/.spyder-py3/userlist.csv', delimiter=",")

userdatapd = userdatapd.drop(columns=['Latitude', 'Longitude', 'IP No.' ])

label_encoder = LabelEncoder()

userdatapd['Country'] = label_encoder.fit_transform(userdatapd['Country'])
userdatapd['AS'] = label_encoder.fit_transform(userdatapd['AS'])
userdatapd['IP Address'] = label_encoder.fit_transform(userdatapd['IP Address'])
userdatapd = userdatapd.head(10)


# selecting 50 different web services

wsdatapd = pd.read_csv('C:/Users/Oumaima/.spyder-py3/wslist.csv', delimiter=",")

wsdatapd = wsdatapd.drop(columns=['Latitude', 'Longitude', 'IP No.'])

label_encoder = LabelEncoder()

wsdatapd['Country'] = label_encoder.fit_transform(wsdatapd['Country'])
wsdatapd['AS'] = label_encoder.fit_transform(wsdatapd['AS'])
wsdatapd['WSDL Address'] = label_encoder.fit_transform(wsdatapd['WSDL Address'])
wsdatapd['Service Provider'] = label_encoder.fit_transform(wsdatapd['Service Provider'])
wsdatapd['IP Address'] = label_encoder.fit_transform(wsdatapd['IP Address'])
wsdatapd = wsdatapd.head(50)


# merging invocations with information about users and web 


qosDatapd = pd.read_csv('C:/Users/Oumaima/.spyder-py3/rtdata.txt', delim_whitespace=True)

userData = userdatapd.to_numpy()
wsData = wsdatapd.to_numpy()
qosData = qosDatapd.to_numpy()


userData = pd.DataFrame(userData, columns=['user id', 'user ip add', 'user country', 'user as' ])

wsData = pd.DataFrame(wsData, columns=['service id', 'service wsdl add', 'service provider', 'service ip add',  'service country', 'service as'])

qosData = pd.DataFrame(qosData, columns=['user id', 'service id', 'time slice id', 'qos value'])

new_qosData = pd.merge(qosData, userData, on='user id')
new_qosData = pd.merge(new_qosData, wsData, on='service id')

new_qosData = new_qosData.reindex(columns = [col for col in new_qosData.columns if col != 'qos value'] + ['qos value'])

new_qosData = new_qosData.loc[new_qosData['time slice id'].isin([i for i in range(64)]),['qos value', 'time slice id','user id','service id','user ip add', 'user country', 'user as', 'service wsdl add', 'service provider', 'service ip add',  'service country', 'service as'] ]




################################################################################################

new_qosData.to_csv('C:/Users/Oumaima/.spyder-py3/sampleFortesting2.csv')
new_qosData = pd.read_csv('C:/Users/Oumaima/.spyder-py3/sampleFortesting.csv',  delimiter=",")
new_qosData = new_qosData.drop(new_qosData.columns[0], axis=1)

################################################################################################


#########################################################################
############################ Generating data ############################
#########################################################################

# generating sequential data

# Step 1: Group the sorted dataset by user ID and service ID

grouped_data = new_qosData.groupby(['time slice id', 'service id'])

sequences =  []
targets = []

# Step 2: Construct sequential representations for each group (user)
for group_name, group_data in grouped_data:
    # Extract features for the sequence
    #sequence_features = group_data[['time slice id', 'service id','qos value', 'user ip add', 'user country', 'user as', 'service wsdl add', 'service provider', 'service ip add',  'service country', 'service as']].values
    sequence_features = group_data[['qos value', 'user ip add', 'user country', 'user as', 'service wsdl add', 'service provider', 'service ip add',  'service country', 'service as']].values
    # Extract target value (response time)
    target = group_data['qos value'].values
    
    
    #if sequence_features.shape == (10, 11):
    if sequence_features.shape == (10, 9):
        sequences.append(sequence_features)
        targets.append(target)


# reshaping sequences and scaling

arr_for_training = np.array(sequences).reshape(np.array(sequences).shape[0]*np.array(sequences).shape[1],np.array(sequences).shape[2])
arr_for_training.astype(float)
    

#La dispersion des valeurs ("variables dispersées" ou "variables écartées".)

scaler = StandardScaler()
scaler = scaler.fit(arr_for_training)
arr_for_training_scaled = scaler.transform(arr_for_training)


#generating my sequences for training

trainX = []
trainY = []

n_future = 1   # Number of service invocations we want to look into the future based on the past ones
n_past = 20  # Number of past invocations we want to use to predict the future

#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my arr_for_training_scaled has a shape (200, 4)
#200 refers to the number of data points and 4 refers to the columns (multi-variables).
for i in range(n_past, len(arr_for_training_scaled) - n_future +1):
    trainX.append(arr_for_training_scaled[i - n_past:i, 0:arr_for_training.shape[1]])
    trainY.append(arr_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

# Split the data into train and test datasets
# test_size = 7995
# trainX_train, trainX_test = trainX[:-test_size], trainX[-test_size:]
# trainY_train, trainY_test = trainY[:-test_size], trainY[-test_size:]


#########################################################################
###################### Building Models & Training #######################
#########################################################################


# define the LSTM model

modelLSTM = Sequential()

modelLSTM.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))

modelLSTM.add(LSTM(32, activation='relu', return_sequences=False))

modelLSTM.add(Dropout(0.2))

modelLSTM.add(Dense(trainY.shape[1]))

modelLSTM.compile(optimizer='adam', loss='mse')
modelLSTM.summary()


# fit the model
historyLSTM = modelLSTM.fit(trainX, trainY, epochs=100, batch_size=64, verbose=1)

#visualisation of loss progress

plt.title('Model Loss - LSTM')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(historyLSTM.history['loss'], label='loss')
plt.legend()


# define the BiLSTM model
modelbilstm = Sequential()

modelbilstm.add(Bidirectional(LSTM(64, activation='relu'), input_shape=(trainX.shape[1], trainX.shape[2])))

modelbilstm.add(Dropout(0.2))

modelbilstm.add(Dense(128, activation='relu'))

modelbilstm.add(Dense(64, activation='relu'))

modelbilstm.add(Dense(32, activation='relu'))

modelbilstm.add(Dense(trainY.shape[1]))  # Assuming you want the output dimension to match trainY.shape[1]

modelbilstm.compile(optimizer='adam', loss='mse')
modelbilstm.summary()

# fit the model
historybilstm = modelbilstm.fit(trainX, trainY, epochs=100, batch_size=64, verbose=1)


#visualisation of loss progress

plt.title('Model Loss - BiLSTM')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(historybilstm.history['loss'], label='loss')
plt.legend()



# define the GRU model

modelGRU = Sequential()
modelGRU.add(GRU(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
modelGRU.add(GRU(32, activation='relu', return_sequences=False))
modelGRU.add(Dropout(0.2))
modelGRU.add(Dense(trainY.shape[1]))

modelGRU.compile(optimizer='adam', loss='mse')
modelGRU.summary()


# fit the model
history = modelGRU.fit(trainX, trainY, epochs=100, batch_size=64, verbose=1)


#visualisation of loss progress

plt.title('Model Loss - GRU')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(history.history['loss'], label='loss')
plt.legend()


#visualisation of loss progress

plt.title('Model Loss - comparison')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(historyLSTM.history['loss'], label='loss LSTM')
plt.plot(history.history['loss'], label='loss GRU')
plt.plot(historybilstm.history['loss'], label='loss BiLSTM')
plt.legend()

#########################################################################
############################## Predictions ##############################
#########################################################################

#Make prediction with LSTM
prediction_LSTM = modelLSTM.predict(trainX[0:1]) #shape = (n, 1) where n is the n_inv_for_prediction
prediction_LSTM

#inverse scaling 

prediction_copies = np.repeat(prediction_LSTM, arr_for_training.shape[1], axis=-1)
y_pred_future_LSTM = scaler.inverse_transform(prediction_copies)[:,0]
y_pred_future_LSTM