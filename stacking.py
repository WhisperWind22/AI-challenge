
# coding: utf-8

# In[3]:


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

#read datasets
train = pd.read_csv('train.csv')
#Removing columns with one value
columnsWithOneValue=[]
for f in train:
    if train[f].value_counts().shape[0]==1:
        columnsWithOneValue=np.append(columnsWithOneValue,[f], axis=0)
train.drop(columnsWithOneValue,axis=1,inplace=True)

X_train, y_train = train.drop('target', 1), train['target']

scaler = QuantileTransformer(output_distribution='normal')
X_train=scaler.fit_transform(X_train)

from sklearn.decomposition import PCA, SparsePCA
pca = PCA(n_components=0.99)
X_train = pca.fit_transform(X_train)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,stratify=y_train, test_size = 0.3, random_state =42)

from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(X_train, y_train)

model = Sequential()

model.add(Dense(256, input_dim=X_res.shape[1]))
model.add(BatchNormalization())
model.add(PReLU())
model.add(Dropout(0.75))
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(PReLU())
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=["accuracy"])

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# check 5 epochs
early_stop = EarlyStopping(monitor='val_acc', patience=5, mode='max') 
callbacks_list = [checkpoint, early_stop]
model.fit(X_res, y_res, batch_size=200, epochs=15, validation_data=(X_test, y_test), verbose=2, 
          callbacks=callbacks_list)

y_true = y_test
y_scores = model.predict(X_test)
scores=roc_auc_score(y_true, y_scores)
print("ROC AUC: %.5f" % (scores))

import lightgbm as lgb
d_train = lgb.Dataset(X_train, y_train)
watchlist = lgb.Dataset(X_test, y_test)
print('Training LGBM model...')
params = {}
params['learning_rate'] = 0.1
params['application'] = 'binary'
params['max_depth'] = 30
params['num_leaves'] = 50
params['verbosity'] = 0
params['metric'] = 'auc'
params['early_stopping_round'] = 40
modelL = lgb.train(params, train_set=d_train, num_boost_round=300, valid_sets=watchlist, 
                  verbose_eval=5)
y_true = y_test
y_scores = modelL.predict(X_test)
scores=roc_auc_score(y_true, y_scores)
print("ROC AUC: %.5f" % (scores))
#roc_auc 0.75

from sklearn.linear_model import Ridge
modelRidge=Ridge()
modelRidge.fit(X_train, y_train)
#roc_auc 0.75
y_true = y_test
y_scores = modelRidge.predict(X_test)
scores=roc_auc_score(y_true, y_scores)
print("Model Ridge")
print("ROC AUC: %.5f" % (scores))

modelTrain=model.predict(X_res)
modelLTrain=modelL.predict(X_res)
modelLTrain.shape=(modelTrain.shape[0],1)
modelRTrain=modelRidge.predict(X_res)
modelRTrain.shape=(modelRTrain.shape[0],1)
X2_train = np.concatenate((modelTrain,modelLTrain,modelRTrain), axis=1)
modelTest=model.predict(X_test)
modelLTest=modelL.predict(X_test)
modelLTest.shape=(modelTest.shape[0],1)
modelRTest=modelRidge.predict(X_test)
modelRTest.shape=(modelRTest.shape[0],1)
X2_test = np.concatenate((modelTest,modelLTest,modelRTest), axis=1)

#layer 2
modelStack = Sequential()

modelStack.add(Dense(128, input_dim=X2_train.shape[1]))
modelStack.add(BatchNormalization())
modelStack.add(PReLU())
modelStack.add(Dropout(0.75))
modelStack.add(Dense(256))
modelStack.add(BatchNormalization())
modelStack.add(PReLU())
modelStack.add(Dropout(0.75))
modelStack.add(Dense(256))
modelStack.add(BatchNormalization())
modelStack.add(PReLU())
modelStack.add(Dropout(0.75))
modelStack.add(Dense(1, activation="sigmoid"))

modelStack.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

modelStack.fit(X2_train, y_res, batch_size=200, epochs=1, validation_data=(X2_test, y_test), verbose=2)
#roc_auc 0.76
y_true = y_test
y_scores = modelStack.predict(X2_test)
scores=roc_auc_score(y_true, y_scores)
print("ROC AUC: %.5f" % (scores))

test = pd.read_csv('test.csv')
test.drop(columnsWithOneValue,axis=1,inplace=True)
Id, X_test = test['Id'], test.drop('Id',1)
X_test=scaler.transform(X_test)
X_test = pca.transform(X_test)
modelTest=model.predict(X_test)
modelLTest=modelL.predict(X_test)
modelLTest.shape=(modelTest.shape[0],1)
modelRTest=modelRidge.predict(X_test)
modelRTest.shape=(modelRTest.shape[0],1)
X2_test = np.concatenate((modelTest,modelLTest,modelRTest), axis=1)
predicted = modelStack.predict(X2_test)

submission = pd.DataFrame()
submission['Id'] = Id
submission['Prediction'] = predicted
submission.to_csv('baseline_submission.csv', index=0)

