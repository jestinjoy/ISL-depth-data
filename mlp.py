#Importing necessary libraries 
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

dataset=np.loadtxt('added.csv',delimiter=",")

data = dataset[:,:-1] #all columns except the last one
target = dataset[:,len(dataset[0])-1] #only the last column
X_train, X_test, y_train, y_test = train_test_split(data, target,test_size=0.3,random_state=109)

# Model Layers Defnition
# Input layer (12 neurons), Output layer (1 neuron)
model = Sequential()
model.add(Dense(80, input_dim=600000, init='uniform', activation='sigmoid'))
model.add(Dense(6, init='uniform', activation='sigmoid'))


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
y_train = keras.utils.to_categorical(y_train, num_classes = 6)
y_test = keras.utils.to_categorical(y_test, num_classes = 6)


model.fit(X_train, y_train, epochs = 300, batch_size = 32)
y_pred = model.predict(X_test)

y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

print(classification_report(y_test_class, y_pred_class))
print(confusion_matrix(y_test_class, y_pred_class))
