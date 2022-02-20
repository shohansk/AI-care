from __future__ import division, print_function

#-----------
from flask import Flask,render_template,request

#import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
#---------------------------

# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from PIL import Image as pil_image
import tensorflow
import keras

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import Model , load_model
from keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

Model= load_model('best_model_cutix.h5')     

class_cancer = {
    0 : 'Actinic keratoses (akiec)',
    1 : 'Basal cell carcinoma (bcc)',
    2 : 'Benign keratosis-like lesions (bkl) ',
    3 : 'Dermatofibroma (df)',
    4 : 'Melanocytic nevi (nv)',
    5 : 'Vascular lesions (vasc)',
    6 : 'Melanoma (mel)'
}

def model_predict(img_path, Model):
    img = image.load_img(img_path, target_size=(28,28,3))


    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    preds = Model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path , Model)

        

        pred_class = preds.argmax(axis=-1)    
        pr = class_cancer[pred_class[0]]
        result =str(pr)         
        return result
    return None

#----------heart parki cancer

@app.route('/liver',methods=['GET','POST'])
def cancer_page():
    if request.method == 'GET':
        return render_template('cancer.html')
    else:
        age = request.form['age']
        sex = request.form['sex']
        total_bilirubin = request.form['chest']
        direct_bilirubin = request.form['trestbps']
        alkaline_phosphotase = request.form['chol']
        alamine_aminotransferase = request.form['fbs']
        aspartate_aminotransferase = request.form['restecg']
        total_proteins = request.form['thalach']
        albumin = request.form['exang']
        albumin_and_globulin_ratio = request.form['oldpeak']

        liver_dataset = pd.read_csv('D:/AI-doctor/liver.csv')
        liver_dataset['Gender'] = liver_dataset['Gender'].map({'Male': 1, 'Female': 2})
        liver_dataset.dropna(inplace=True)
        X = liver_dataset.drop(columns='Dataset', axis=1)
        Y = liver_dataset['Dataset']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=101)
        model1 = RandomForestClassifier(n_estimators = 100)
        model1.fit(X_train, Y_train)
        input_data = (age,sex,total_bilirubin,direct_bilirubin,alkaline_phosphotase,alamine_aminotransferase,aspartate_aminotransferase,total_proteins,albumin,albumin_and_globulin_ratio)
        input_data_as_numpy_array= np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = model1.predict(input_data_reshaped)
        senddata=""
        if (prediction[0]== 2):
            senddata='According to the given details person does not have Liver Disease'
        else:
            senddata='According to the given details chances of having Liver Disease are High, So Please Consult a Doctor'
        return render_template('result.html',resultvalue=senddata)

@app.route('/heart',methods=['GET','POST'])
def heart_page():
    if request.method == 'GET':
        return render_template('heart.html')
    else:
        age = request.form['age']
        sex = request.form['sex']
        chest = request.form['chest']
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        oldpeak = request.form['oldpeak']
        slope = request.form['slope']
        ca = request.form['ca']
        thal = request.form['thal']
        heart_dataset = pd.read_csv('D:/AI-doctor/heart.csv')
        X = heart_dataset.drop(columns='target', axis=1)
        Y = heart_dataset['target']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=101)
        model1 = LogisticRegression()
        model1.fit(X_train, Y_train)
        input_data = (age,sex,chest,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
        input_data_as_numpy_array= np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = model1.predict(input_data_reshaped)
        senddata=""
        if (prediction[0]== 0):
            senddata='According to the given details person does not have Heart Disease'
        else:
            senddata='According to the given details chances of having Heart Disease are High, So Please Consult a Doctor'
        return render_template('result.html',resultvalue=senddata)

@app.route('/parki',methods=['GET','POST'])
def parki_page():
    if request.method == 'GET':
        return render_template('parki.html')
    else:
        Fo = request.form['Fo']
        Fhi= request.form['Fhi']
        Flo= request.form['Flo']
        Jitter = request.form['Jitter']
        Jitterabs= request.form['Jitterabs']
        RAP = request.form['RAP']
        PPQ = request.form['PPQ']
        DDP = request.form['DDP']
        Shimmer = request.form['Shimmer']
        Shimmerdb = request.form['Shimmerdb']
        APQ3 = request.form['APQ3']
        APQ5 = request.form['APQ5']
        APQ = request.form['APQ']
        DDA = request.form['DDA']
        NHR = request.form['NHR']
        HNR = request.form['HNR']
        RPDE = request.form['RPDE']
        DFA = request.form['DFA']
        Spread1 = request.form['Spread1']
        Spread2 = request.form['Spread2']
        D2 = request.form['D2']
        PPE = request.form['PPE']
        parkinsons_dataset = pd.read_csv('D:/AI-doctor/parkinsons.data')
        X = parkinsons_dataset.drop(columns=['name','status'], axis=1)
        Y = parkinsons_dataset['status']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=101)
       # model1 = LogisticRegression()
        model1 = svm.SVC(kernel='linear')
        model1.fit(X_train, Y_train)
        input_data = (Fo,Fhi,Flo,Jitter,Jitterabs,RAP,PPQ,DDP,Shimmer,Shimmerdb,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,Spread1,Spread2,D2,PPE)
        input_data_as_numpy_array= np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = model1.predict(input_data_reshaped)
        senddata=""
        if (prediction[0]== 0):
            senddata='According to the given details person does not have parkinsons Disease'
        else:
            senddata='According to the given details chances of having parkinsons are High, So Please Consult a Doctor'
        return render_template('result.html',resultvalue=senddata)

#------------------------------------------


if __name__ == '__main__':
    app.run(debug=False) # Production Environment, if debug=True == Development Environment
    
