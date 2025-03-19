from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            LOAN=request.form.get('LOAN'),
            MORTDUE=float(request.form.get('MORTDUE')),
            VALUE=float(request.form.get('VALUE')),
            REASON=request.form.get('REASON'),
            JOB=request.form.get('JOB'),
            YOJ=float(request.form.get('YOJ')),
            DEROG=float(request.form.get('DEROG')),
            DELINQ=float(request.form.get('DELINQ')),
            CLAGE=float(request.form.get('CLAGE')),
            NINQ=float(request.form.get('NINQ')),
            CLNO=float(request.form.get('CLNO')),
            DEBTINC=float(request.form.get('DEBTINC'))
            )
        
        # Prepare the data for prediction
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        # Get the prediction result from the model
        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("After Prediction")

        # Convert the prediction result into a human-readable message
        if results[0] == 1:
            result_message = "You will most likely default on the loan."
        else:
            result_message = "You will most likely not default on the loan."
        
        # Render the result message on the web page
        return render_template('home.html', result_message=result_message)

if __name__=="__main__":
    app.run(host="0.0.0.0")        