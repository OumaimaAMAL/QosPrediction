# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 10:39:15 2023

@author: Oumaima
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, LSTM, BatchNormalization
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dropout


# selecting 100 different users

userdatapd = pd.read_csv('C:/Users/Oumaima/.spyder-py3/userlist.csv', delimiter=",")

userdatapd = userdatapd.drop(columns=['Latitude', 'Longitude', 'IP No.' ])

label_encoder = LabelEncoder()

userdatapd['Country'] = label_encoder.fit_transform(userdatapd['Country'])
userdatapd['AS'] = label_encoder.fit_transform(userdatapd['AS'])
userdatapd['IP Address'] = label_encoder.fit_transform(userdatapd['IP Address'])
userdatapd = userdatapd.head(100)


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


############################################################################"
# saving my new data 

new_qosData.to_csv('C:/Users/Oumaima/.spyder-py3/sampleFortesting.csv')
new_qosData = pd.read_csv('C:/Users/Oumaima/.spyder-py3/sampleFortesting.csv',  delimiter=",")
new_qosData = new_qosData.drop(new_qosData.columns[0], axis=1)

#############################################################################"


# generating sequential data

# Step 1: Group the sorted dataset by user ID and service ID

grouped_data = new_qosData.groupby(['service id','user id'])

sequences =  []
targets = []

# Step 2: Construct sequential representations for each group (user)
for group_name, group_data in grouped_data:
    # Extract features for the sequence
    sequence_features = group_data[['qos value', 'time slice id','user ip add', 'user country', 'user as', 'service wsdl add', 'service provider', 'service ip add',  'service country', 'service as']].values
    # Extract target value (response time)
    target = group_data['qos value'].values

    if sequence_features.shape == (64, 10):
        sequences.append(sequence_features)
        targets.append(target)


# reshaping sequences and scaling

arr_for_training = np.array(sequences).reshape(np.array(sequences).shape[0]*np.array(sequences).shape[1],np.array(sequences).shape[2])
arr_for_training.astype(float)
    
scaler = StandardScaler()
scaler = scaler.fit(arr_for_training)
arr_for_training_scaled = scaler.transform(arr_for_training)


# Number of past invocations we want to use to predict the future 63 : n_past-1
n_past = 64

# Initialize trainX and trainY
trainX = []
trainY = []

# Loop through the array to create sequences
for i in range(0, len(arr_for_training)-n_past+1, n_past):
    
    past_sequence = arr_for_training_scaled[i:i+n_past-1, :]
    
    # First value of the next row is the value to predict
    
    future_value = arr_for_training_scaled[i+n_past-1:i+n_past,0]  
    trainX.append(past_sequence)
    trainY.append(future_value)


trainX, trainY = np.array(trainX), np.array(trainY)


# define the LSTM model

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()

# fit the model
history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_split=0.2, verbose=1)


#visualisation of loss progress

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()





