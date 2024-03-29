import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application
## import ridge regressor model and standard scaler pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        # Get values from form
        Temprature = float(request.form.get('Temprature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        
        # Handle the case where 'Classes' field is empty or not submitted
        Classes = float(request.form.get('Classes')) if request.form.get('Classes') else 0.0
        
        # Handle the case where 'Region' field is empty or not submitted
        Region = float(request.form.get('Region')) if request.form.get('Region') else 0.0

        new_data_scaled = standard_scaler.transform([[Temprature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html', result=result[0])
       
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
