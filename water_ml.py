import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('water_potability.csv')
df['Potability']=df['Potability'].astype('category')

#Preprocessing the data
#Replacing NaN values in PH column
phMean_0 = df[df['Potability'] == 0]['ph'].mean(skipna=True) #skipping NaN values in calc mean of col values
df.loc[(df['Potability'] == 0) & (df['ph'].isna()), 'ph'] = phMean_0
phMean_1 = df[df['Potability'] == 1]['ph'].mean(skipna=True)
df.loc[(df['Potability'] == 1) & (df['ph'].isna()), 'ph'] = phMean_1

#Replacing NaN values in Sulfate Column
SulfateMean_0 = df[df['Potability'] == 0]['Sulfate'].mean(skipna=True)
df.loc[(df['Potability'] == 0) & (df['Sulfate'].isna()), 'Sulfate'] = SulfateMean_0
SulfateMean_1 = df[df['Potability'] == 1]['Sulfate'].mean(skipna=True)
df.loc[(df['Potability'] == 1) & (df['Sulfate'].isna()), 'Sulfate'] = SulfateMean_1

#Replacing NaN values in Trihalomethanes column
TrihalomethanesMean_0 = df[df['Potability'] == 0]['Trihalomethanes'].mean(skipna=True)
df.loc[(df['Potability'] == 0) & (df['Trihalomethanes'].isna()), 'Trihalomethanes'] = TrihalomethanesMean_0
TrihalomethanesMean_1 = df[df['Potability'] == 1]['Trihalomethanes'].mean(skipna=True)
df.loc[(df['Potability'] == 1) & (df['Trihalomethanes'].isna()), 'Trihalomethanes'] = TrihalomethanesMean_1

#Preparing the data - Loading the data into X and y

X = df.drop('Potability', axis = 1).copy()
y = df['Potability'].copy()

from sklearn.model_selection import (GridSearchCV, KFold, train_test_split, cross_val_score)
#Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)

#Balancing the data
#Synthetic Oversampling
from imblearn.over_sampling import SMOTE
from collections import Counter
print('Balancing the data by SMOTE - Oversampling of Minority level\n')
smt = SMOTE()
counter = Counter(y_train)
print('Before SMOTE', counter)
X_train, y_train = smt.fit_resample(X_train, y_train)
counter = Counter(y_train)
print('\nAfter SMOTE', counter)

#Normalizing the Data
from sklearn.preprocessing import StandardScaler
ssc = StandardScaler()

X_train = ssc.fit_transform(X_train)
X_test = ssc.transform(X_test)

#random forest classifier
from sklearn.ensemble import RandomForestClassifier

model =RandomForestClassifier()
model.fit(X_train,y_train)

ypred=model.predict(X_test)

#Evaluating the model

output={'y_pred':ypred,'y_actual':y_test}
output=pd.DataFrame(output)
print(output)
accuracy_random=model.score(X_test, y_test)
#accuracies of the model
print('Training Accuracy :', model.score(X_train, y_train))
print('Testing Accuracy :', accuracy_random )

#New prediction
#inputt = [float(x) for x in "8.316765884214679 214.37339408562252 22018.417440775294 8.05933237743854 356.88613564305666 363.2665161642437 18.436524495493302 100.34167436508008 4.628770536837084".split(' ')]
#inputt=[float(x) for x in "4.668101687405915 193.68173547507868 47580.99160333534 7.166638935482532 359.94857436696 526.4241709223593 13.894418518194527 66.68769478539706 4.4358209095098".split(' ')]
#inputt=[float(x) for x in "7.119824384264552,156.70499334039215,18730.813653342713,3.6060360905057203,282.3440504739606,347.71502726194376,15.929535908825699,79.5007783369744,3.445756223321899".split(',')]
inputt=[float(x) for x in "5.667650646643197,229.9283665401693,16953.89873630931,8.77430610207418,293.5742499094306,554.1205361936269,14.254640735422015,54.4367023778621,3.633213756490487".split(',')]
print(inputt)

final = [np.array(inputt)]
print(final)
final=ssc.transform(final)
print(final)
prediction=model.predict(final)
print(prediction)

pickle.dump(model,open('water_model.pkl','wb'))
#Pickling - generates pickle file

model=pickle.load(open('water_model.pkl','rb'))
print("SUcess loaded")