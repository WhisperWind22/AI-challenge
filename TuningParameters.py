
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from sklearn.preprocessing import QuantileTransformer
from keras.layers.advanced_activations import PReLU
from sklearn.metrics import roc_auc_score
from keras.callbacks import EarlyStopping,ModelCheckpoint
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp

#read datasets
train = pd.read_csv('train.csv')
#Removing columns with one value
columnsWithOneValue=[]
for f in train:
    if train[f].value_counts().shape[0]==1:
        columnsWithOneValue=np.append(columnsWithOneValue,[f], axis=0)
train.drop(columnsWithOneValue,axis=1,inplace=True)

X_train, y_train = train.drop('target', 1), train['target']
y_train = y_train.as_matrix()
Y_train=y_train

scaler = QuantileTransformer(output_distribution='normal')
X_train=scaler.fit_transform(X_train)

from sklearn.decomposition import PCA, SparsePCA
pca = PCA(n_components=0.99)
X_train = pca.fit_transform(X_train)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,stratify=y_train, 
                                                    test_size = 0.3, random_state =42)

from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(X_train, y_train)

def objective(params):
    print ('Params testing: ', params)
    
    model = Sequential()
    model.add(Dense(params['units1'], input_dim = X_res.shape[1])) 
    model.add(PReLU())
    model.add(Dropout(params['dropout1']))
    model.add(Dense(params['units2'])) 
    model.add(PReLU())
    model.add(Dropout(params['dropout2']))
    if params['choice']['layers']== 'three':
        model.add(Dense(params['choice']['units3'])) 
        model.add(PReLU())
        model.add(Dropout(params['choice']['dropout3']))
        model.add(BatchNormalization())
        patience=10
    else:
        patience=5
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'], metrics=["accuracy"])
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # check 5 epochs
    early_stop = EarlyStopping(monitor='val_acc', patience=patience, mode='max') 
    callbacks_list = [checkpoint, early_stop]
    model.fit(X_res, y_res, epochs=params['epochs'], batch_size=params['batch_size'], 
              validation_data=(X_test, y_test), verbose = 0,callbacks=callbacks_list)
    y_true = y_test
    y_scores = model.predict(X_test)

    score=roc_auc_score(y_true, y_scores)
    print('Test ROC AUC:', score)
    print ('\n ')
    return {'loss':1-score,'roc_auc': score, 'status': STATUS_OK, 'model': model}

space = {'choice':
hp.choice('num_layers',
    [
                    {'layers':'two',
                     
                                                    
                    },
        
                     {'layers':'three',
                      
                      
                      'units3': hp.choice('units3', [ 256, 1024]),
                      'dropout3': hp.choice('dropout3', [0.25,0.5,0.75])
                                
                    }
        
    
    ]),
    
    'units1': hp.choice('units1', [ 256, 512,]),
    'units2': hp.choice('units2', [ 1024,2048]),
                 
    'dropout1': hp.choice('dropout1', [0.5,0.75]),
    'dropout2': hp.choice('dropout2', [0.25,0.5,0.75]),
    
    'batch_size' : hp.choice('batch_size', [64,200]),
         
    'epochs' :  40,
         
    'optimizer': 'nadam',
    
    }

trials = Trials()

best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=100)

print (best)
print (trials.best_trial)

