import numpy as np
import pandas as pd
from split import splitData
from p1 import omitOutliers
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt
from keras import optimizers
from keras.callbacks import LambdaCallback


def dataWrapper():

    dataSet = splitData(df = omitOutliers())
    trainSet = dataSet['trainSet']
    testSet = dataSet['testSet']

    trainSetX = trainSet.loc[:,'mcg':'nuc']
    trainSetClass = trainSet['Class']

    testSetX = testSet.loc[:,'mcg':'nuc']
    testSetClass = testSet['Class']

    return {'trainX':trainSetX, 'trainClass':trainSetClass,'testX':testSetX,'testClass':testSetClass}

def feedFoward():
    data = dataWrapper()

    training_data = np.array(data['trainX'])  #xtrain
    training_target = np.array(data['trainClass']) #ytrain

    testing_data = np.array(data['testX']) #xtest
    testing_target = np.array(data['testClass']) #ytest

    dummies_train = np.array(pd.get_dummies(training_target)) #ytrain dummies
    dummies_test = np.array(pd.get_dummies(testing_target))  #dummy variables of testing set labels


    #To create each layers
    activation_1 = layers.Dense(units=3, activation='sigmoid') #First layer
    activation_2 = layers.Dense(units=3, activation='sigmoid') #Second layer
    output_layer = layers.Dense(units=9,activation='softmax') #Output layer

    #To build up the model
    model = Sequential([activation_1,activation_2,output_layer]) #initialized the model

    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])


    weight_receive= []
    print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: (weight_receive.append(output_layer.get_weights())))

    history = model.fit(training_data, dummies_train, epochs=10,batch_size=1,verbose=0,validation_data=[testing_data,dummies_test],
             callbacks = [print_weights])


    cyt_weight = []
    for epoch in weight_receive:
        ww = epoch[0] #weights
        bb = epoch[1] #bias

        ss = np.append(bb[0], ww[:,0]) #weights and bias for class cyt

        cyt_weight.append(ss)

    cyt_weight = np.vstack(cyt_weight)


    plt.plot(cyt_weight[:,0])
    plt.plot(cyt_weight[:,1])
    plt.plot(cyt_weight[:,2])
    plt.plot(cyt_weight[:,3])
    plt.title('Wegihts')
    plt.ylabel('weights')
    plt.xlabel('epoch')
    plt.legend(['weights', 'weight2','weight3','bias'],loc='upper right')
    plt.show()


    error_tr = [1 - acc for acc in history.history['acc']]
    error_te = [1 - acc for acc in history.history['val_acc']]

    print('Training errors:',error_tr,'\n')
    print('Testing errors:',error_te)

    plt.plot(error_te)
    plt.plot(error_tr)
    plt.title('model error')
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


#feedFoward()
