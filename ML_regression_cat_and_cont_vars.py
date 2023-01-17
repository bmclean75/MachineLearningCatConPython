#for a helpful tutorial see
#https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/

# mlp for regression
# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from keras.optimizers import SGD
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow import keras
from keras.callbacks import LearningRateScheduler
import numpy as np
import pandas as pd
import math #sqrt

# load the dataset
path = 'combined_dataset.csv'
dataset = read_csv(path,header=0)

# shape
print(dataset.shape)

# descriptions
print(dataset.describe())

print("LotArea minimum = ", dataset["LotArea"].min())
print("LotArea maximum = ", dataset["LotArea"].max())
print("LotArea mean = ", dataset["LotArea"].mean())

print("SalePrice minimum = ", dataset["SalePrice"].min())
print("SalePrice maximum = ", dataset["SalePrice"].max())
print("SalePrice mean = ", dataset["SalePrice"].mean())


#select values [minXrow:maxXrow,minYcol:maxYcol]
#row:col indexing, starting from 0...
X1 = dataset.values[:,4:5] #LotArea
print("X1 = ",X1)

#convert categorical variables into one-hot variables
#https://stackoverflow.com/questions/34007308/linear-regression-analysis-with-string-categorical-features-variables
X2 = pd.get_dummies(dataset["MSZoning"].values)
#X2 = dataset.values[:,2:3]
print("X2 = ",X2)

X3 = dataset.values[:,17:19] #OverallQuality, OverallCondition
print("X3 = ",X3)

X = np.concatenate((X1,X2,X3),axis=1)
y = dataset.values[:,80:81]
print(X)
print(y)
print("X = ",X)


# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train = np.asarray(X_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

# determine the number of input features
n_features = X_train.shape[1]
# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
#model.add(BatchNormalization()) #https://keras.io/api/layers/normalization_layers/batch_normalization/
model.add(Dropout(0.5)) #https://keras.io/api/layers/regularization_layers/dropout/

model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.5))

model.add(Dense(1))

#output activation functions:
#for Regression: linear (because values are unbounded) - this is default
#for Classification: softmax (simple sigmoid works too but softmax works better)
#Use simple sigmoid if your output admits multiple "true" answers, for instance, 
#a network that checks for the presence of various objects in an image. In other words, 
#the output is not a probability distribution (does not need to sum to 1).
#Use softmax if probability sums to 1.

# compile the model
#******************
#optimizers = RMSprop, Adagrad, or Adam... 
#https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/
model.compile(optimizer='adam', loss='mse')

# fit the model
#https://www.tensorflow.org/api_docs/python/tf/keras/Model
history = model.fit(X_train, y_train, epochs=1000, batch_size=1, verbose=0, use_multiprocessing=True)

# evaluate the model
error = model.evaluate(X_test, y_test, verbose=0)
print('MSE: %.3f, RMSE: %.3f' % (error, math.sqrt(error)))
y_pred = model.predict(X_test)
MAE = mean_absolute_error(y_test, y_pred)
print('mean abs error: %.3f' % (MAE))
# make a prediction

row = [10000,1,0,0,0,0,2,2] #dummyvar[C (all),FV,RH,RL,RM],LotArea
yhat = model.predict([row])
print('Predicted: %.3f' % yhat)
row = [10000,0,1,0,0,0,4,4] #dummyvar 1,2,3,4,5,LotArea
yhat = model.predict([row])
print('Predicted: %.3f' % yhat)
row = [10000,0,0,1,0,0,6,6] #dummyvar 1,2,3,4,5,LotArea
yhat = model.predict([row])
print('Predicted: %.3f' % yhat)
row = [10000,0,0,0,1,0,8,8] #dummyvar 1,2,3,4,5,LotArea
yhat = model.predict([row])
print('Predicted: %.3f' % yhat)
row = [10000,0,0,0,0,1,10,10] #dummyvar 1,2,3,4,5,LotArea
yhat = model.predict([row])
print('Predicted: %.3f' % yhat)

#PLOTS
#https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# list all data in history
print(history.history.keys())

#no history for accuracy because regression

#https://stackoverflow.com/questions/41908379/keras-plot-training-validation-and-test-set-accuracy
# summarize history for loss
pyplot.plot(history.history['loss'])
#pyplot.plot(history.history['val_loss'])
pyplot.title('model loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper left')
pyplot.show()