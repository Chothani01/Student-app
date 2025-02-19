import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import numpy as np
import streamlit as st
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://princechothani53:HHplNTJ9feCPD2005@cluster0.7vcpe.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["Student"] # "Student is name of db"
collection = db["student_pred"] # "student_pred" is name of collection which is in db


def load_model():
    with open("Student_pred_Trained_Model.pkl", "rb") as f:
        model, scaler, le = pickle.load(f)
    return model, scaler, le

def preprocessing_input_data(data, scaler, le):
    data["Extracurricular Activities"] = le.transform([data["Extracurricular Activities"]])
    df = pd.DataFrame([data])
    data_transformed = scaler.transform(df)
    return data_transformed 
    
def predict_data(data):
    model, scaler, le = load_model()
    processed_data = preprocessing_input_data(data, scaler, le)
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.title("Student performance prediction")
    st.write("Enter your details for get prediction on your input")
    
    hour_studied = st.number_input("Hour studied", min_value=1, max_value=10, value=5)
    previous_score = st.number_input("Previous Scores", min_value=35, max_value=100, value=50)
    extracurricular_Activities = st.selectbox("Extracurricular activities", ["Yes", "No"])
    sleep_hours = st.number_input("Sleep Hours", min_value=3, max_value=24, value=7)
    question_papers = st.number_input("Question papers", min_value=0, max_value=10, value=5)
    
    if st.button("Predict_Your_Score"):
        user_data = {
            "Hours Studied":hour_studied,
            "Previous Scores":previous_score,
            "Extracurricular Activities":extracurricular_Activities,
            "Sleep Hours":sleep_hours,
            "Sample Question Papers Practiced":question_papers
        }
    
        prediction = predict_data(user_data)
        st.success(f"Your prediction result is: {prediction}")
        
        user_data["Prediction"] = prediction # add new dictionary in user_name dictionary
        user_data = {
                        key: int(value) if isinstance(value, (np.integer, np.int32, np.int64)) else 
                            float(value) if isinstance(value, (np.floating, np.float32, np.float64, np.ndarray)) else 
                            value 
                            for key, value in user_data.items()
                    } # convert numpydatatype to normal python data type because mongodb support normal python datatype

        collection.insert_one(user_data)
        
if __name__=="__main__":
    main()
