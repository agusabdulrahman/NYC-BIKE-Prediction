import pandas as pd
import numpy as np 
import joblib
import streamlit

model = open("../Lineaer Regresion/experiment/linear_regresion_model.pkl", "rb")
lr_model = joblib.load(model)

def lr_prediction(HighTemp, LowTemp, Precipitation):
    pred_arr = np.array([HighTemp, LowTemp, Precipitation])
    preds = pred_arr.reshape(1, -1)
    preds = preds.astype(float)
    model_prediction = lr_model.predict(preds)
    return model_prediction

def run():
    streamlit.title("Predict NYC-BIKE")
    html_temp= """
    """
    streamlit.markdown(html_temp)
    High_Temp = streamlit.number_input("High Temp")
    Low_Temp = streamlit.number_input("Low Temp")
    Precipitation = streamlit.number_input("Precipitation")
    
    prediction = ""
    formatted_prediction = ""
    
    if streamlit.button("Predict"):
        prediction = lr_prediction(High_Temp, Low_Temp, Precipitation)
        formatted_prediction = "{:.3f}".format(prediction[0])
    streamlit.success("The Total predicted NYC-BIKE Prediction is: {}".format(formatted_prediction))

if __name__ == '__main__':
    run()
        