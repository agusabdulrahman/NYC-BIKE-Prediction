import pandas as pd
import numpy as np
import sklearn
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the pre-trained model
model = open("../experiment/linear_regresion_model.pkl", "rb")
lr_model = joblib.load(model)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    model_prediction = None  # Initial default value
    if request.method == 'POST':
        print(request.form.get('HighTemp'))
        print(request.form.get('LowTemp'))
        print(request.form.get('Precipitation'))
        try:
            HighTemp = float(request.form['HighTemp'])
            LowTemp = float(request.form['LowTemp'])
            Precipitation = float(request.form['Precipitation'])
            pred_args = [HighTemp, LowTemp, Precipitation]
            pred_arr = np.array(pred_args)
            # preds = pred_arr.reshape()
            model_prediction = lr_model.predict([pred_arr])
            model_prediction = round(float(model_prediction[0]), 2)
        except ValueError:
            return "Please enter valid values"
    return render_template('predict.html', prediction=model_prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0')