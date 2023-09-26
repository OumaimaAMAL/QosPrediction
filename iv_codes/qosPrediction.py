# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:48:42 2023

@author: Oumaima
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout


#20 users &  10 services

new_qosData = pd.read_csv('C:/Users/Oumaima/.spyder-py3/sample.csv',  delimiter=",")
new_qosData = new_qosData.drop(new_qosData.columns[0], axis=1)
new_qosData

# Step 1: Group the sorted dataset by user ID
grouped_data = new_qosData.groupby(['service id','user id'])

sequences =  []
targets = []

# Step 2: Construct sequential representations for each group (user)
for group_name, group_data in grouped_data:
    # Extract features for the sequence
    sequence_features = group_data[['qos value','time slice id','user ip add', 'user ip no', 'user as', 'service wsdl add' ]].values
    # Extract target value (response time)
    target = group_data['qos value'].values

    sequences.append(sequence_features)
    targets.append(target)
    

    
arr_for_training = np.array(sequences).reshape(np.array(sequences).shape[0]*np.array(sequences).shape[1],np.array(sequences).shape[2])
arr_for_training.astype(float)
arr_for_training


#scaling my data :
    
scaler = StandardScaler()
scaler = scaler.fit(arr_for_training)
arr_for_training_scaled = scaler.transform(arr_for_training)
arr_for_training_scaled

#generating my sequences for training

# Number of past invocations we want to use to predict the future
n_past = 64

# Initialize trainX and trainY
trainX = []
trainY = []

# Loop through the array to create sequences
for i in range(0, len(arr_for_training)-n_past+1, n_past):
    past_sequence = arr_for_training[i:i+n_past-1, :]
    future_value = arr_for_training[i+n_past-1:i+n_past, 0]  # First value of the next row is the value to predict
    trainX.append(past_sequence)
    trainY.append(future_value)


trainX, trainY = np.array(trainX), np.array(trainY)


# Number of past invocations we want to use to predict the future
n_past = 64

# Initialize trainX and trainY
trainX_s = []
trainY_s = []

# Loop through the array to create sequences
for i in range(0, len(arr_for_training)-n_past+1, n_past):
    past_sequence = arr_for_training_scaled[i:i+n_past-1, :]
    future_value = arr_for_training_scaled[i+n_past-1:i+n_past,0]  # First value of the next row is the value to predict
    trainX_s.append(past_sequence)
    trainY_s.append(future_value)


trainX_s, trainY_s = np.array(trainX_s), np.array(trainY_s)



# define the LSTM model

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX_s.shape[1], trainX_s.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY_s.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()


# fit the model
history = model.fit(trainX_s, trainY_s, epochs=100, batch_size=16, validation_split=0.1, verbose=1)



plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()


n_inv_for_prediction = 50

#Make prediction
prediction = model.predict(trainX[-n_inv_for_prediction:]) #shape = (n, 1) where n is the n_inv_for_prediction
prediction

#inverse scaling 

prediction_copies = np.repeat(prediction, arr_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]
y_pred_future


#selecting original qos values of last service invocations in time slice = 0
y_original = pd.DataFrame(np.array(sequences)[4])[0]
np.array(y_original)



# Define the time indices for the values
time_indices = range(len(y_pred_future))

# Plot the predicted values
plt.plot(time_indices, y_pred_future, label='Predicted')

# Plot the original values
plt.plot(time_indices, y_original, label='Original')

# Set the axis labels and title
plt.xlabel('Time Index')
plt.ylabel('Value')
plt.title('Comparison of Predicted and Original Values')

# Add a legend
plt.legend()

# Show the plot
plt.show()


loss = model.evaluate(trainX, trainY)
loss



