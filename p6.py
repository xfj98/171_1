import numpy as np
import pandas as pd
from keras.models import Sequential
from keras import layers
from keras import optimizers
from p2 import dataWrapper


def predict():
    data = dataWrapper()

    training_data = np.array(data['trainX']) #xtrain
    training_target = np.array(data['trainClass']) #ytrain

    testing_data = np.array(data['testX']) #xtest
    testing_target = np.array(data['testClass']) #ytest

    dummies_train = np.array(pd.get_dummies(training_target)) #ytrain dummies
    dummies_test = np.array(pd.get_dummies(testing_target))  #dummy variables of testing set labels
    lenClass = len(dummies_test[0])

    #To create each layers
    activation_1 = layers.Dense(units=3, activation='sigmoid') #First layer
    activation_2 = layers.Dense(units=3, activation='sigmoid') #Second layer
    output_layer = layers.Dense(units=lenClass,activation='softmax') #Output layer

    #To build up the model
    model = Sequential([activation_1,activation_2,output_layer]) #initialized the model

    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    history = model.fit(training_data, dummies_train, epochs=50,batch_size=1, verbose=0, validation_data=[testing_data,dummies_test])

    x = np.asarray([0.52, 0.47, 0.52, 0.23, 0.55, 0.03, 0.52, 0.39])
    x_trans = np.transpose([x])
    y = model.predict(np.transpose(x_trans))
    print('predicted probabilities for each class:','\n',y)

#predict()
