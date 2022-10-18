
from tensorflow.keras.layers import (MaxPooling1D,
    Input, add, Dropout, BatchNormalization,LayerNormalization, GlobalAveragePooling1D, MaxPooling1D,
    TimeDistributed, Activation, Add, SimpleRNN , Bidirectional, LSTM, Flatten, Dense, Conv1D)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential


num_classes = 5
output_activation_fn = 'sigmoid' if num_classes == 1 else 'softmax'

def simpleSequentialModel(X_train, config):

  model = Sequential()
  
  model.add(Dense(64, input_shape=(X_train.shape[1],)))
  model.add(Activation('relu'))
  model.add(Dropout(rate=config['dropout']))

  model.add(Dense(32))
  model.add(Activation('relu'))
  model.add(Dropout(rate=config['dropout']))

  model.add(Dense(16))
  model.add(Activation('relu'))
  model.add(Dropout(rate=config['dropout']))

  model.add(Dense(num_classes,activation=output_activation_fn))

  return compileModel(model, config['learning_rate'])


def simple1D_CNN(X_train, config):

  
  model = Sequential()
  model.add(Conv1D(filters = 128, kernel_size = 5, activation = 'relu', input_shape = (X_train.shape[1],X_train.shape[2])))
  model.add(Dropout(rate=config['dropout']))
  model.add(Flatten())
  model.add(Dense(num_classes,activation=output_activation_fn))

  return  compileModel(model,config['learning_rate'])

def CNN_model(X_train, config):
  model= Sequential(name='CNN')
  model.add(Conv1D(filters = 256, kernel_size = 3, strides = 1, activation = 'relu', input_shape = (X_train.shape[1],X_train.shape[2])))
  model.add(Conv1D(filters = 256, kernel_size = 2, strides = 1, activation = 'relu'))
  model.add(Conv1D(filters = 256, kernel_size = 2, strides = 1, activation = 'relu'))
  model.add(BatchNormalization())
  model.add(Dropout(rate=config['dropout']))
  model.add(Conv1D(filters = 256, kernel_size = 2, strides = 1, activation = 'relu'))
  model.add(Conv1D(filters = 256, kernel_size = 2, strides = 1, activation = 'relu'))
  model.add(Conv1D(filters = 256, kernel_size = 2, strides = 1, activation = 'relu'))
  model.add(BatchNormalization())
  model.add(Dropout(rate=config['dropout']))
  model.add(GlobalAveragePooling1D())
  model.add(Dense(100, activation='relu'))
  model.add(Dense(num_classes,activation=output_activation_fn))

  return  compileModel(model,config['learning_rate'])

def CNN_Ach(X_train, config):
  model= Sequential(name='CNN_Ach')
  model.add(Conv1D(filters = 64, kernel_size = 3, strides = 1, activation = 'relu', input_shape = (X_train.shape[1],X_train.shape[2])))
  model.add(BatchNormalization())
  model.add(Dropout(rate=config['dropout']))
  model.add(Conv1D(filters = 128, kernel_size = 2, strides = 1, activation = 'relu'))
  model.add(BatchNormalization())
  model.add(Dropout(rate=config['dropout']))
  model.add(Conv1D(filters = 256, kernel_size = 2, strides = 1, activation = 'relu'))
  model.add(BatchNormalization())
  model.add(Dropout(rate=config['dropout']))
  model.add(GlobalAveragePooling1D())
  model.add(Dense(100, activation='relu'))
  model.add(Dense(num_classes,activation=output_activation_fn))


def LSTM_model(X_train, config):

  model = Sequential(name = 'LSTM')

  model.add(LSTM(100,return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2])))
  model.add(Dropout(rate=config['dropout']))

  model.add(LSTM(100,return_sequences=True))
  model.add(Dropout(rate=config['dropout']))

  model.add(LSTM(100,return_sequences=True))
  model.add(Dropout(rate=config['dropout']))

  model.add(Flatten())
  model.add(Dense(100, activation='relu'))
  model.add(Dropout(rate=config['dropout']))

  model.add(Dense(num_classes,activation=output_activation_fn))

  return compileModel(model,config['learning_rate'])




def compileModel(model, lr):

 
  opt = Adam(lr)
  #TODO: Double check sparse
  loss = 'binary_crossentropy' if num_classes==1 else 'sparse_categorical_crossentropy'

  model.compile(
  loss = loss,
  optimizer=opt, metrics=['accuracy'])
  

  return model

def generateModel(X_train, config):

  model = None

  
  if(config['model_choice']=="simple"):
          
    model = simpleSequentialModel(X_train, config)

 
  if(config['model_choice']=="1D_CNN"):
  
    model = simple1D_CNN(X_train, config)

  if(config['model_choice']=="LSTM"):

    model = LSTM_model(X_train, config)

  if(config['model_choice']=="CNN"):

    model = CNN_model(X_train, config)
  if(config['model_choice']=="CNN_Ach"):

    model = CNN_model(X_train, config)

  
  
  return model