import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

import get_LA_crime_data as gcd

import h5py
import time

# Train a deep, fully-connected neural network that, given the month, the time of day, age, sex, descent, and location, outputs the probability of each type of crime occurring

print('Please wait while the program runs!')

NUM_CLASSES = 11

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def get_model(input_shape, NUM_CLASSES):

    X_input = Input((input_shape,))
    
    # First hidden layer(s)
    X = Dense(units = 8, activation = 'relu')(X_input)
    # Hidden layer feature normalization
    X = BatchNormalization()(X)
    # For regularization
    X = Dropout(0.2)(X)

    # Second hidden layer(s)
    X = Dense(units = 9, activation = 'relu')(X)
    X = BatchNormalization()(X)
    X = Dropout(0.2)(X)

    # Third hidden layer(s)
    X = Dense(units = 10, activation = 'relu')(X)
    X = BatchNormalization()(X)
    X = Dropout(0.2)(X)

    # Output layer
    X = Dense(units = NUM_CLASSES, activation = 'softmax')(X)

    # Create model
    model = Model(inputs = X_input, outputs = X)

    return model

# Only include the first time running, to collect the training data from the online dataset
# gcd.get_features()

train_dataset = h5py.File('train.h5', "r")
X_train = np.array(train_dataset["data_X"][:]) # your train set features
Y_train = np.array(train_dataset["data_Y"][:]) # your train set labels
Y_train = Y_train.reshape((1, Y_train.shape[0]))

# Normalize X_train (remember: one feature per column!)
X_train = (X_train - np.mean(X_train, axis = 0)) / np.std(X_train, axis = 0)
Y_train = convert_to_one_hot(Y_train, NUM_CLASSES)
print(X_train.shape, Y_train.shape)

model = get_model(X_train.shape[0], NUM_CLASSES)
model.compile(optimizer = Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

checkpointer = ModelCheckpoint(filepath='./logs/weights-improvement-{epoch:02d}-{val_acc:.2f}.h5', monitor='val_acc', save_best_only=True, save_weights_only=True, period=1)
tensorboard = TensorBoard(log_dir='./logs/{}'.format(time.time()))

model.fit(X_train.T, Y_train.T, epochs = 20000, batch_size = X_train.shape[1], callbacks = [checkpointer, tensorboard], validation_split = 0.1)

model.save('model.h5')

