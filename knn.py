#Importing necessary libraries 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn import metrics
#Reading the data
dataset=np.loadtxt('added.csv',delimiter=",")



knn=KNeighborsClassifier(n_neighbors=2)


#Splitting the data  into target and data
data = dataset[:,:-1] #all columns except the last one

target = dataset[:,len(dataset[0])-1] #only the last column

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(data, target,test_size=0.3,random_state=109)


#Model Training
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)


#Model Evaluation
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred,average="macro"))
print(metrics.precision_recall_fscore_support(y_test, y_pred, average="macro"))
print(metrics.confusion_matrix(y_test, y_pred))
