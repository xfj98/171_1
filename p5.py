from p2 import dataWrapper
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras import optimizers
from keras import layers

data = dataWrapper()

training_data = np.array(data['trainX'])  #xtrain
training_target = np.array(data['trainClass']) #ytrain

testing_data = np.array(data['testX']) #xtest
testing_target = np.array(data['testClass']) #ytest

dummies_train = np.array(pd.get_dummies(training_target)) #ytrain dummies
dummies_test = np.array(pd.get_dummies(testing_target))  #dummy variables of testing set labels

def lay1node3():

    #To create each layers
    activation_1 = layers.Dense(units=3, activation='sigmoid') #First layer
    output_layer = layers.Dense(units=9,activation='softmax') #Output layer

    #To build up the model
    model = Sequential([activation_1,output_layer]) #initialized the model

    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    history = model.fit(training_data, dummies_train, epochs=50,batch_size=1,verbose=0,validation_data=[testing_data,dummies_test])
    test_result = model.evaluate(testing_data, dummies_test, batch_size = 1, verbose = 0, )
    test_error = (1-(test_result[1]))

    print('testing errors for 1 layer and 3 nodes:',test_error)

def lay1node6():

    #To create each layers
    activation_1 = layers.Dense(units=6, activation='sigmoid') #First layer
    output_layer = layers.Dense(units=9,activation='softmax') #Output layer

    #To build up the model
    model = Sequential([activation_1,output_layer]) #initialized the model

    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    history = model.fit(training_data, dummies_train, epochs=50,batch_size=1,verbose=0,validation_data=[testing_data,dummies_test])
    test_result = model.evaluate(testing_data, dummies_test, batch_size = 1, verbose = 0, )
    test_error = (1-(test_result[1]))

    print('testing errors for 1 layer 6 nodes:',test_error)


def lay1node9():
    #To create each layers
    activation_1 = layers.Dense(units=9, activation='sigmoid') #First layer
    output_layer = layers.Dense(units=9,activation='softmax') #Output layer

    #To build up the model
    model = Sequential([activation_1,output_layer]) #initialized the model

    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    history = model.fit(training_data, dummies_train, epochs=50,batch_size=1,verbose=0,validation_data=[testing_data,dummies_test])
    test_result = model.evaluate(testing_data, dummies_test, batch_size = 1, verbose = 0, )
    test_error = (1-(test_result[1]))

    print('testing errors for 1 layer 9 nodes:',test_error)

def lay1node12():
    #To create each layers
    activation_1 = layers.Dense(units=12, activation='sigmoid') #First layer
    output_layer = layers.Dense(units=9,activation='softmax') #Output layer

    #To build up the model
    model = Sequential([activation_1,output_layer]) #initialized the model

    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    history = model.fit(training_data, dummies_train, epochs=50,batch_size=1,verbose=0,validation_data=[testing_data,dummies_test])
    test_result = model.evaluate(testing_data, dummies_test, batch_size = 1, verbose = 0, )
    test_error = (1-(test_result[1]))

    print('testing errors for 1 layer and 12 nodes:',test_error)

def lay2node3():
    activation_1 = layers.Dense(units=3, activation='sigmoid') #First layer
    activation_2 = layers.Dense(units=3, activation='sigmoid') #Second layer
    output_layer = layers.Dense(units=9,activation='softmax') #Output layer

    #To build up the model
    model = Sequential([activation_1,activation_2,output_layer]) #initialized the model

    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    history = model.fit(training_data, dummies_train, epochs=50,batch_size=1,verbose=0,validation_data=[testing_data,dummies_test])
    test_result = model.evaluate(testing_data, dummies_test, batch_size = 1, verbose = 0, )
    test_error = (1-(test_result[1]))

    print('testing errors for 2 layers and 3 nodes:',test_error)


def lay2node6():
    activation_1 = layers.Dense(units=6, activation='sigmoid') #First layer
    activation_2 = layers.Dense(units=6, activation='sigmoid') #Second layer
    output_layer = layers.Dense(units=9,activation='softmax') #Output layer

    #To build up the model
    model = Sequential([activation_1,activation_2,output_layer]) #initialized the model

    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    history = model.fit(training_data, dummies_train, epochs=50,batch_size=1,verbose=0,validation_data=[testing_data,dummies_test])
    test_result = model.evaluate(testing_data, dummies_test, batch_size = 1, verbose = 0, )
    test_error = (1-(test_result[1]))

    print('testing errors for 2 layers and 6 nodes:',test_error)

def lay2node9():
    activation_1 = layers.Dense(units=9, activation='sigmoid') #First layer
    activation_2 = layers.Dense(units=9, activation='sigmoid') #Second layer
    output_layer = layers.Dense(units=9,activation='softmax') #Output layer

    #To build up the model
    model = Sequential([activation_1,activation_2,output_layer]) #initialized the model

    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    history = model.fit(training_data, dummies_train, epochs=50,batch_size=1,verbose=0,validation_data=[testing_data,dummies_test])
    test_result = model.evaluate(testing_data, dummies_test, batch_size = 1, verbose = 0, )
    test_error = (1-(test_result[1]))

    print('testing errors for 2 layers and 9 nodes:',test_error)

def lay2node12():

    activation_1 = layers.Dense(units=12, activation='sigmoid') #First layer
    activation_2 = layers.Dense(units=12, activation='sigmoid') #Second layer
    output_layer = layers.Dense(units=9,activation='softmax') #Output layer

    #To build up the model
    model = Sequential([activation_1,activation_2,output_layer]) #initialized the model

    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    history = model.fit(training_data, dummies_train, epochs=50,batch_size=1,verbose=0,validation_data=[testing_data,dummies_test])
    test_result = model.evaluate(testing_data, dummies_test, batch_size = 1, verbose = 0, )
    test_error = (1-(test_result[1]))

    print('testing errors for 2 layers and 12 nodes:',test_error)

def lay3node3():
    activation_1 = layers.Dense(units=3, activation='sigmoid') #First layer
    activation_2 = layers.Dense(units=3, activation='sigmoid') #Second layer
    activation_3 = layers.Dense(units=3, activation='sigmoid') #Third layer
    output_layer = layers.Dense(units=9,activation='softmax') #Output layer

    #To build up the model
    model = Sequential([activation_1,activation_2,activation_3,output_layer]) #initialized the model

    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    history = model.fit(training_data, dummies_train, epochs=50,batch_size=1,verbose=0,validation_data=[testing_data,dummies_test])
    test_result = model.evaluate(testing_data, dummies_test, batch_size = 1, verbose = 0, )
    test_error = (1-(test_result[1]))

    print('testing errors for 3 layers and 3 nodes:',test_error)

def lay3node6():
    activation_1 = layers.Dense(units=6, activation='sigmoid') #First layer
    activation_2 = layers.Dense(units=6, activation='sigmoid') #Second layer
    activation_3 = layers.Dense(units=6, activation='sigmoid') #Third layer
    output_layer = layers.Dense(units=9,activation='softmax') #Output layer

    #To build up the model
    model = Sequential([activation_1,activation_2,activation_3,output_layer]) #initialized the model

    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    history = model.fit(training_data, dummies_train, epochs=50,batch_size=1,verbose=0,validation_data=[testing_data,dummies_test])
    test_result = model.evaluate(testing_data, dummies_test, batch_size = 1, verbose = 0, )
    test_error = (1-(test_result[1]))

    print('testing errors for 3 layers and 6 nodes:',test_error)

def lay3node9():
    activation_1 = layers.Dense(units=9, activation='sigmoid') #First layer
    activation_2 = layers.Dense(units=9, activation='sigmoid') #Second layer
    activation_3 = layers.Dense(units=9, activation='sigmoid') #Third layer
    output_layer = layers.Dense(units=9,activation='softmax') #Output layer

    #To build up the model
    model = Sequential([activation_1,activation_2,activation_3,output_layer]) #initialized the model

    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    history = model.fit(training_data, dummies_train, epochs=50,batch_size=1,verbose=0,validation_data=[testing_data,dummies_test])
    test_result = model.evaluate(testing_data, dummies_test, batch_size = 1, verbose = 0, )
    test_error = (1-(test_result[1]))

    print('testing errors for 3 layers and 9 nodes:',test_error)

def lay3node12():
    activation_1 = layers.Dense(units=12, activation='sigmoid') #First layer
    activation_2 = layers.Dense(units=12, activation='sigmoid') #Second layer
    activation_3 = layers.Dense(units=12, activation='sigmoid') #Third layer
    output_layer = layers.Dense(units=9,activation='softmax') #Output layer

    #To build up the model
    model = Sequential([activation_1,activation_2,activation_3,output_layer]) #initialized the model

    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    history = model.fit(training_data, dummies_train, epochs=50,batch_size=1,verbose=0,validation_data=[testing_data,dummies_test])
    test_result = model.evaluate(testing_data, dummies_test, batch_size = 1, verbose = 0, )
    test_error = (1-(test_result[1]))

    print('testing errors for 3 layers and 12 nodes:',test_error)

# lay1node3()
# lay1node6()
# lay1node9()
# lay1node12()
# lay2node3()
# lay2node6()
# lay2node9()
# lay2node12()
# lay3node3()
# lay3node6()
# lay3node9()
# lay3node12()
