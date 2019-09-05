from keras.models import Sequential
from keras import layers
from keras import optimizers
from p1 import omitOutliers
import numpy as np
import pandas as pd
from p2 import dataWrapper

def weightUpdate():

    data = omitOutliers()

    xtrain = np.array(data.loc[:,'mcg':'nuc'])
    ytrain = np.array(pd.get_dummies(data['Class']))
    xtrain1 = pd.DataFrame(xtrain)
    xtrain1 = xtrain1.iloc[0:1,]
    ytrain1 = pd.DataFrame(ytrain)
    ytrain1 = ytrain1.iloc[0:1,]


    activation_1 = layers.Dense(units=3,input_dim=8,activation='sigmoid',kernel_initializer='zeros',bias_initializer='zeros')
    activation_2 = layers.Dense(units=3,activation='sigmoid',kernel_initializer='ones',bias_initializer='zeros')
    outout_Layer = layers.Dense(units=9,activation='softmax',kernel_initializer='ones',bias_initializer='zeros')

    model = Sequential([activation_1,activation_2,outout_Layer])

    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    history = model.fit(xtrain1,ytrain1,epochs=5,batch_size=1,verbose=1)


    weights_1 = activation_2.get_weights()[0]
    weights_2 = outout_Layer.get_weights()[0]
    print('ypredicted:','\n',ytrain1)
    print(weights_1)
    print(weights_2)

# weightUpdate()
