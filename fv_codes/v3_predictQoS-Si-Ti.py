# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 23:22:53 2023

@author: Oumaima
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Dense, LSTM, Bidirectional, Input, concatenate
from keras.layers import Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


new_qosData = pd.read_csv('D:/France/.spyder-py3/sampleFortesting2.csv',  delimiter=",")
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
    sequence_features = group_data[['time slice id', 'qos value', 'user ip add', 'user country', 'user as', 'service wsdl add', 'service ip add',  'service country', 'service as']].values
    #sequence_features = group_data[['qos value', 'user ip add', 'user country', 'user as', 'service wsdl add', 'service ip add',  'service country', 'service as']].values
    # Extract target value (response time)
    target = group_data['qos value'].values
    
    
    if sequence_features.shape == (10, 9):
    #if sequence_features.shape == (10, 8):
        sequences.append(sequence_features)
        targets.append(target)


# reshaping sequences and scaling

arr_for_training = np.array(sequences).reshape(np.array(sequences).shape[0]*np.array(sequences).shape[1],np.array(sequences).shape[2])
arr_for_training.astype(float)
    

#La dispersion des valeurs ("varia bles dispersées" ou "variables écartées".)

scaler = StandardScaler()
scaler = scaler.fit(arr_for_training)
arr_for_training_scaled = scaler.transform(arr_for_training)


#generating training time slices for the corresponding sequences of trainX
train_time = arr_for_training_scaled[20:32000, 0:1]
train_time.shape


#generating training tasks for the corresponding sequences of trainX
train_task = arr_for_training_scaled[20:32000, [2,3,4]]
train_task.shape


#generating my sequences for training
arr_for_training_scaled = arr_for_training_scaled[:, [1, 2, 5, 6, 7, 8]]


#arr_for_training_scaled = arr_for_training_scaled[:, 1:]

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
test_size = 2500
trainX_train, trainX_test = trainX[:-test_size], trainX[-test_size:]
train_time_train, train_time_test = train_time[:-test_size], train_time[-test_size:]
train_task_train, train_task_test = train_task[:-test_size], train_task[-test_size:]
trainY_train, trainY_test = trainY[:-test_size], trainY[-test_size:]

#########################################################################
###################### Building Models & Training #######################
#########################################################################


# Assuming trainX, train_task, and train_time are my data arrays for QoS, task, and time information

# Input layer for the sequence of QoS values in the past
input_qos = Input(shape=(trainX.shape[1], trainX.shape[2]))

# Input layer for the task information
input_task = Input(shape=(train_task.shape[1],))  # Assuming train_task has shape (31980, 3)

# Input layer for the time information (which time slice)
input_time = Input(shape=(train_time.shape[1],))  # Assuming train_time has shape (31980, 1)

# BiLSTM layer for the QoS values
lstm_layer = Bidirectional(LSTM(64, activation='relu'))(input_qos)

# Concatenate the LSTM output with the task and time inputs
concat_layer = concatenate([lstm_layer, input_time, input_task])

# Additional Dense layers as before
dropout_layer = Dropout(0.4)(concat_layer)
dense_layer1 = Dense(128, activation='relu')(dropout_layer)
dense_layer2 = Dense(64, activation='relu')(dense_layer1)
dense_layer3 = Dense(32, activation='relu')(dense_layer2)

# Output layer
output_layer = Dense(trainY.shape[1])(dense_layer3)

# Create the model with three inputs and one output
modelBiLSTM = Model(inputs=[input_qos, input_time, input_task], outputs=output_layer)

optimizer = Adam(learning_rate=0.001)
modelBiLSTM.compile(optimizer=optimizer, loss='mse')
modelBiLSTM.summary()


# Fit the model with two inputs
historybilstm = modelBiLSTM.fit(
    [trainX, train_time, train_task], 
    trainY, 
    epochs=100, 
    batch_size=64, 
    verbose=1)


# Fit the model with two inputs
historybilstmvalidation = modelBiLSTM.fit([trainX, train_time, train_task], trainY, epochs=100, batch_size=64, verbose=1)


#visualisation of loss progress

plt.title('Model Loss - BiLSTM')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(historybilstm.history['loss'], label='loss')
plt.legend()

#visualisation of loss progress

plt.title('Model Loss - BiLSTM')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(historybilstmvalidation.history['loss'], label='loss')
plt.legend()

#version2.1

# Define input layers
input_qos = Input(shape=(trainX.shape[1], trainX.shape[2]))
input_task = Input(shape=(train_task.shape[1],))
input_time = Input(shape=(train_time.shape[1],))

# BiLSTM layer for the QoS values
lstm_layer = Bidirectional(LSTM(64, activation='relu'))(input_qos)

# Concatenate the LSTM output with the task and time inputs
concat_layer = concatenate([lstm_layer, input_time, input_task])

# Additional Dense layers
dense_layer1 = Dense(128, activation='relu')(concat_layer)
dropout_layer = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(64, activation='relu')(dropout_layer)
dense_layer3 = Dense(32, activation='relu')(dense_layer2)

# Output layer
output_layer = Dense(trainY.shape[1])(dense_layer3)

# Create the model with three inputs and one output
modelBiLSTM = Model(inputs=[input_qos, input_time, input_task], outputs=output_layer)

# Compile the model
optimizer = Adam(learning_rate=0.0001)
modelBiLSTM.compile(optimizer=optimizer, loss='mse')
modelBiLSTM.summary()

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model with validation data
history = modelBiLSTM.fit(
    [trainX_train, train_time_train, train_task_train],
    trainY_train,
    epochs=100,
    batch_size=64,
    validation_data=([trainX_test, train_time_test, train_task_test], trainY_test),
    callbacks=[early_stopping],
    verbose=1
)

#visualisation of loss progress

plt.title('Model Loss - BiLSTM')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()


#model evaluation

score = modelBiLSTM.evaluate([trainX, train_time, train_task], trainY, verbose=1)

# Visualize the model
plot_model(modelBiLSTM, to_file='C:/Users/Oumaima/.spyder-py3/modelBiLSTM.png', show_shapes=True, show_layer_names=True)


# Display the model diagram
img = plt.imread('C:/Users/Oumaima/.spyder-py3/modelBiLSTM.png')
plt.imshow(img)
plt.axis('off')
plt.show()


#########################################################################


#Make prediction with LSTM
prediction_LSTM = modelBiLSTM.predict([trainX_test, train_time_test, train_task_test]) #shape = (n, 1) where n is the n_inv_for_prediction
prediction_LSTM

#inverse scaling

prediction_copies = np.repeat(prediction_LSTM, arr_for_training.shape[1], axis=-1)
y_pred_future_LSTM = scaler.inverse_transform(prediction_copies)[:,0]
y_pred_future_LSTM


#original values
trainY_test
#inverse scaling

prediction_copies = np.repeat(trainY_test, arr_for_training.shape[1], axis=-1)
y_original = scaler.inverse_transform(prediction_copies)[:,0]
y_original


# Define the time indices for the values
#sample_indices = range(len(y_original))
sample_indices = range(200)

# Plot the predicted values by BiLSTM model
plt.plot(sample_indices, y_pred_future_LSTM[:200], label='BiLSTM')

# Plot the original values
plt.plot(sample_indices, y_original[:200], label='Original')

# Set the axis labels and title
plt.xlabel('Sample Index ')
plt.ylabel('Value')
plt.title('Comparison of Predicted and Original Values')

# Add a legend
plt.legend()

# Show the plot
plt.show()



############################## Predictions ##############################
#########################################################################


#Make prediction with LSTM
prediction_LSTM = modelBiLSTM.predict(trainX[0:1]) #shape = (n, 1) where n is the n_inv_for_prediction
prediction_LSTM

#inverse scaling 

prediction_copies = np.repeat(prediction_LSTM, arr_for_training.shape[1], axis=-1)
y_pred_future_LSTM = scaler.inverse_transform(prediction_copies)[:,0]
y_pred_future_LSTM
