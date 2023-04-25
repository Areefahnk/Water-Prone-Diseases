from flask import Flask,request,render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('water_model.pkl','rb')) #water potability ML model



#Water Potability Model --------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (GridSearchCV, KFold, train_test_split, cross_val_score)

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
#Water Potability Model ------------------------------------------------------------------------------------------------
from keras.models import load_model
# Model saved with Keras model.save()
MODEL_PATH = 'model_vgg19.h5'

# Load your trained model
quality_model = load_model(MODEL_PATH)
#prediction = quality_model.predict([[1,2,3,4,5,6,7]])
#print(prediction)
@app.route('/')
def hello_world():
    return render_template("home.html")

@app.route('/waterquality.html', methods=['GET'])
def qualitywebpage():
    # R

    return render_template('waterquality.html')

@app.route('/waterpotabilitytest.html', methods=['GET'])
def water_potabilitytest():
    # R

    return render_template('waterpotabilitytest.html')

@app.route('/quality',methods=['POST','GET'])
def quality():
    float_features = [float(x) for x in request.form.values()]
    #final = [np.array(float_features)]
    print(float_features)
    #print(final)
    prediction = quality_model.predict([float_features])
    print(prediction)
    if prediction[0][0]<=45:
        return render_template("waterquality.html", pred='Water Quality is Good',
                               inp='Predicted for inputs:\nDissolved Oxygen(mg/l): %.2f, PH: %.2f, Conductivity: %.1f, Biochemical Oxygen Demand (BOD - mg/l): %.1f, Total Nitrogen (mg/l): %.1f, Fecal Coliform (mg/l): %.1f, Total Coliform (mg/l): %.1f' % (
                                   float_features[0], float_features[1], float_features[2], float_features[3],
                                   float_features[4], float_features[5], float_features[6]))

    elif prediction[0][0]<=71:
        return render_template("waterquality.html", pred='Water Quality is Fair enough',
                               inp='Predicted for inputs:\nDissolved Oxygen(mg/l): %.2f, PH: %.2f, Conductivity: %.1f, Biochemical Oxygen Demand (BOD - mg/l): %.1f, Total Nitrogen (mg/l): %.1f, Fecal Coliform (mg/l): %.1f, Total Coliform (mg/l): %.1f' % (
                                   float_features[0], float_features[1], float_features[2], float_features[3],
                                   float_features[4], float_features[5], float_features[6]))

    else:
        return render_template("waterquality.html", pred='Water Quality is Poor',
                           inp='Predicted for inputs:\nDissolved Oxygen(mg/l): %.2f, PH: %.2f, Conductivity: %.1f, Biochemical Oxygen Demand (BOD - mg/l): %.1f, Total Nitrogen (mg/l): %.1f, Fecal Coliform (mg/l): %.1f, Total Coliform (mg/l): %.1f' % (
                           float_features[0], float_features[1], float_features[2], float_features[3],float_features[4],float_features[5],float_features[6]))


@app.route('/predict',methods=['POST','GET'])
def predict():
    float_features=[float(x) for x in request.form.values()]
    final=[np.array(float_features)]
    print(float_features)
    print(final) #inputs in 2D array format
    #-------------------------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    # Balancing the data
    # Synthetic Oversampling
    from imblearn.over_sampling import SMOTE
    from collections import Counter
    print('Balancing the data by SMOTE - Oversampling of Minority level\n')
    smt = SMOTE()
    counter = Counter(y_train)
    print('Before SMOTE', counter)
    X_train, y_train = smt.fit_resample(X_train, y_train)
    counter = Counter(y_train)
    print('\nAfter SMOTE', counter)
    # Normalizing the Data
    from sklearn.preprocessing import StandardScaler
    ssc = StandardScaler()
    X_train = ssc.fit_transform(X_train) #finalizing scale
    X_test = ssc.transform(X_test)
    # -------------------------------------------------------------------------------------------
    #doing standard scaling on the inputs given by user
    final = ssc.transform(final)
    print(final)
    prediction = model.predict(final)
    print(prediction)
    #Potability: Indicates if water is safe
    #for human consumption.Potable - 1 and Not potable - 0
    if prediction[0]==0:
        return render_template('waterpotabilitytest.html',
                               pred='Result from Water potability test: Water is Not Potable to drink',
                               inp='Predicted for inputs:\nPH: %.2f, Hardness: %.2f, Solids: %.1f, Chloramines: %.1f, Sulfate: %.1f, Conductivity: %.1f, Organic Carbon: %.1f, Trihalomethanes: %.1f, Turbidity: %.1f' % (
                                   float_features[0], float_features[1], float_features[2], float_features[3],
                                   float_features[4], float_features[5], float_features[6], float_features[7],
                                   float_features[8]))
    else:
        return render_template('waterpotabilitytest.html',
                           pred='Result from Water potability test: Water is Potable to drink',
                           inp='Predicted for inputs:\nPH: %.2f, Hardness: %.2f, Solids: %.1f, Chloramines: %.1f, Sulfate: %.1f, Conductivity: %.1f, Organic Carbon: %.1f, Trihalomethanes: %.1f, Turbidity: %.1f' % (
                           float_features[0], float_features[1], float_features[2], float_features[3],float_features[4],float_features[5],float_features[6],float_features[7],float_features[8]))

if __name__ == '__main__':
    app.run(debug=True)
