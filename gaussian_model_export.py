import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
#import pickle

#Reading the data
#csvr=csv.reader(open('dataset.csv','r'))
#x=list(csvr)
#dataset=np.array(x)

dataset=np.loadtxt('added.csv',delimiter=",")


#Splitting the data  into target and data
data = dataset[:,:-1] #all columns except the last one
target = dataset[:,len(dataset[0])-1] #only the last column
#data = data.astype(np.float64)
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3,random_state=109)
#Model Training
gnb = GaussianNB()
model = gnb.fit(X_train, y_train)
y_pred=model.predict(X_test)
# save
#with open('model.pkl','wb') as f:
#    pickle.dump(gnb,f)

#Model Evaluation
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred,average="micro"))
print(metrics.precision_recall_fscore_support(y_test, y_pred, average="macro"))
print(metrics.confusion_matrix(y_test, y_pred))
