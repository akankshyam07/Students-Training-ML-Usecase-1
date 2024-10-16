"""
Author: Akankshya Mohanty
Mentor & Reviewer: Rajani Vanarse
#*******************************************************************
#Copyright (C) 2023 Adino Labs
#*******************************************************************
"""
#importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

#creating dataset
train_dataset = pd.read_csv('KDDTrain+.csv');
test_dataset = pd.read_csv('KDDTest+.csv');

#Renaming the columns
train_dataset.columns = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','aa','ab','ac','ad','ae','af','ag','ah','ai','aj','ak','al','am','an','ao','ap','aq']
test_dataset.columns = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','aa','ab','ac','ad','ae','af','ag','ah','ai','aj','ak','al','am','an','ao','ap','aq']
print('Shape of the training dataset:' + str(train_dataset.shape))
x = train_dataset.iloc[:,0:42].values
y = train_dataset.iloc[:,42].values
x_test = test_dataset.iloc[:,0:42].values
y_test = test_dataset.iloc[:,42].values

#creating dependent variable class for result
factor = pd.factorize(train_dataset['ap'])
train_dataset.ap = factor[0]
definitions = factor[1]
#print(train_dataset.ap.head())
print(definitions)

factor_test = pd.factorize(test_dataset['ap'])
test_dataset.ap = factor_test[0]
definitions_test = factor_test[1]
print(definitions_test)

#splitting data into independent and dependent variables
x = train_dataset.iloc[:,0:42].values
y = train_dataset.iloc[:,42].values
x_test = test_dataset.iloc[:,0:42].values
y_test = test_dataset.iloc[:,42].values

#encoding categorial data

"""
labelencoder_x_1 = LabelEncoder()
labelencoder_x_2 = LabelEncoder()
labelencoder_x_3 = LabelEncoder()
x[:,1] = labelencoder_x_1.fit_transform(x[:,1])
x[:,2] = labelencoder_x_2.fit_transform(x[:,2])
x[:,3] = labelencoder_x_3.fit_transform(x[:,3])


onehotencoder_1 = OneHotEncoder(categorical_features = [1])
x = onehotencoder_1.fit_transform(x).toarray()
onehotencoder_2 = OneHotEncoder(categorical_features = [4])
x = onehotencoder_2.fit_transform(x).toarray()
onehotencoder_3 = OneHotEncoder(categorical_features = [70])
x = onehotencoder_3.fit_transform(x).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

print('x'+str(x))
print('y'+str(y))

"""

labelencoder_x_1 = LabelEncoder()
labelencoder_x_2 = LabelEncoder()
labelencoder_x_3 = LabelEncoder()
x[:,1] = labelencoder_x_1.fit_transform(x[:,1])
x[:,2] = labelencoder_x_2.fit_transform(x[:,2])
x[:,3] = labelencoder_x_3.fit_transform(x[:,3])

print('x1' + str(x[:,1]))
print('x2' + str(x[:,2]))
print('x3' + str(x[:,3]))

x_test[:,1] = labelencoder_x_1.fit_transform(x_test[:,1])
x_test[:,2] = labelencoder_x_2.fit_transform(x_test[:,2])
x_test[:,3] = labelencoder_x_3.fit_transform(x_test[:,3])
"""
cols_to_transform = ['b','c','d']
df_with_dummies = pd.get_dummies( columns = cols_to_transform)
"""

#defining train and test data 
x_train = x
y_train = y

#feature scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#creating Random Forest classifier and training
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(x_train, y_train)

#predicting test results
y_pred = classifier.predict(x_test)

#print accuracy
print("Accuracy of Random Forest Classifier in % :  " + str(accuracy_score(y_test,y_pred)*100))

#creating confusion matrix
cm = confusion_matrix(y_test,y_pred)
#print(cm)