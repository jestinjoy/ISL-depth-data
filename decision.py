#Importing necessary libraries 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

dataset=np.loadtxt('added.csv',delimiter=",")

data = dataset[:,:-1] #all columns except the last one
target = dataset[:,len(dataset[0])-1] #only the last column
X_train, X_test, y_train, y_test = train_test_split(data, target,test_size=0.3,random_state=109)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

#Model Evaluation
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred,average="macro"))
print(metrics.precision_recall_fscore_support(y_test, y_pred, average="macro"))
print(metrics.confusion_matrix(y_test, y_pred))
