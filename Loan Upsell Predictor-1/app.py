"""
@author:Sagar Sinha

"""
from flask import Flask, render_template, url_for, request, jsonify, redirect
import pandas as pd
import numpy as np
import joblib 


app = Flask(__name__)


#Load the models

xgbr_model = joblib.load("B:/ML-NLP/Projects/lTFsdeploy/lTFs-finance/saved models/xgbr.pkl")
lgbm_model = joblib.load("B:/ML-NLP/Projects/lTFsdeploy/lTFs-finance/saved models/lgbm.pkl")
rmr_model = joblib.load("B:/ML-NLP/Projects/lTFsdeploy/lTFs-finance/saved models/rmr.pkl")
print("Models Loaded")
    
#Use flask_cors or secret key




@app.route("/", methods=['GET']) #This signifies the default route
@app.route("/home")
def home():
    return render_template("index.html", title="Home Page")


#Run the app from main method
if __name__ == '__main__':
   app.run(debug=True)