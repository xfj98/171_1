import numpy as np
import pandas as pd
from split import splitData
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt
from keras import optimizers
from load import loadData
from keras.callbacks import LambdaCallback

def setData():

    data = loadData()

    dataSet = splitData(data)
    trainSet = dataSet['trainSet']
    testSet = dataSet['testSet']

    xtrain = np.array(trainSet.loc[:,'mcg':'nuc'])
    ytrain = np.array(trainSet['Class'])
    ytrain = np.array(pd.get_dummies(ytrain))
    #print(ytrain)
    #print(len(ytrain[0]))

    xtest = np.array(testSet.loc[:,'mcg':'nuc'])
    ytest = np.array(testSet['Class'])
    ytest = np.array(pd.get_dummies(ytest))

    activation_1 = layers.Dense(units=3, activation='sigmoid') #First layer
    activation_2 = layers.Dense(units=3, activation='sigmoid') #Second layer
    output_layer = layers.Dense(units=10,activation='softmax') #Output layer

    #To build up the model
    model = Sequential([activation_1,activation_2,output_layer]) #initialized the model
    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    weight_receive= []
    print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: ([weight_receive.append(output_layer.get_weights())]).append(activation_2.get_weights()))


    history = model.fit(xtrain, ytrain, epochs=50,batch_size=1,verbose=0,callbacks = [print_weights])

    error = []
    for i in range(len(history.history['acc'])):
        error.append(1-(history.history['acc'][i]))

    output_weight = output_layer.get_weights()
    layer2 = activation_2.get_weights()

    print('Training error:',error[len(error)-1])
    print('Outputlayer all weights:','\n',output_weight[0])
    print('Outputlayer bias:','\n',output_weight[1])
    print('Second Hidden layer all weights:','\n',layer2[0])
    print('Second Hidden layer bias:','\n',layer2[1])



#setData()
