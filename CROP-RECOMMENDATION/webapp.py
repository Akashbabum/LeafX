## Importing necessary libraries for the web app
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Display Images
from PIL import Image
import requests
from io import BytesIO

def load_image_from_url(url, timeout=5):
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
        'Accept': 'image/*'
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

try:
    # List of backup image URLs
    image_urls = [
        "https://i.ibb.co/V0zKXccx/crop.png",
        "https://images.unsplash.com/photo-1574943320219-553eb213f72d",
        "https://images.pexels.com/photos/2286202/pexels-photo-2286202.jpeg"
    ]
    
    # Try each URL until one works
    img = None
    for url in image_urls:
        try:
            img = load_image_from_url(url)
            break
        except:
            continue
    
    # If an image was loaded successfully, display it
    if img:
        st.image(img, caption="Welcome to Crop Recommendation System", use_container_width=True)
    else:
        raise Exception("Failed to load image from all sources")
    
except Exception as e:
    st.warning("Unable to load welcome image, continuing without it...")
    st.markdown("## Welcome to Crop Recommendation System")

# Load data from CSV file
try:
    # First try to load from local file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'Crop_recommendation.csv')
    
    st.info("Loading dataset...")
    
    # Try local file first
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.success("Dataset loaded successfully from local file!")
    else:
        # Fallback to remote file
        csv_url = "https://raw.githubusercontent.com/akashbabum/LeafX/main/CROP-RECOMMENDATION/Crop_recommendation.csv"
        df = pd.read_csv(csv_url)
        st.success("Dataset loaded successfully from remote source!")
    
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")
    st.info("Please ensure the dataset file is available")
    st.stop()

#features = df[['temperature', 'humidity', 'ph', 'rainfall']]
X = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']
labels = df['label']

# Split the data into training and testing sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)
RF = RandomForestClassifier(n_estimators=20, random_state=5)
RF.fit(Xtrain,Ytrain)
predicted_values = RF.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)


# Function to load and display an image of the predicted crop
def show_crop_image(crop_name):
    # Assuming we have a directory named 'crop_images' with images named as 'crop_name.jpg'
    image_path = os.path.join('crop_images', crop_name.lower()+'.jpg')
    if os.path.exists(image_path):
        st.image(image_path, caption=f"Recommended crop: {crop_name}", use_column_width=True)
    else:
        st.error("Image not found for the predicted crop.")


import pickle
# Dump the trained Naive Bayes classifier with Pickle
RF_pkl_filename = 'RF.pkl'
# Open the file to save as pkl file
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
# Close the pickle instances
RF_Model_pkl.close()


#model = pickle.load(open('RF.pkl', 'rb'))
RF_Model_pkl=pickle.load(open('RF.pkl','rb'))

## Function to make predictions
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    # # Making predictions using the model
    prediction = RF_Model_pkl.predict(np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1))
    return prediction

## Streamlit code for the web app interface
def main():  
    # # Setting the title of the web app
    st.markdown("<h1 style='text-align: center;'>SMART CROP RECOMMENDATIONS", unsafe_allow_html=True)
    
    st.sidebar.title("LeafX")
    # # Input fields for the user to enter the environmental factors
    st.sidebar.header("Enter Crop Details")
    nitrogen = st.sidebar.number_input("Nitrogen", min_value=0.0, max_value=140.0, value=0.0, step=0.1)
    phosphorus = st.sidebar.number_input("Phosphorus", min_value=0.0, max_value=145.0, value=0.0, step=0.1)
    potassium = st.sidebar.number_input("Potassium", min_value=0.0, max_value=205.0, value=0.0, step=0.1)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=51.0, value=0.0, step=0.1)
    humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    ph = st.sidebar.number_input("pH Level", min_value=0.0, max_value=14.0, value=0.0, step=0.1)
    rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)
    inputs=[[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]]                                               
   
    # # Validate inputs and make prediction
    inputs = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    if st.sidebar.button("Predict"):
        if not inputs.any() or np.isnan(inputs).any() or (inputs == 0).all():
            st.error("Please fill in all input fields with valid values before predicting.")
        else:
            prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
            st.success(f"The recommended crop is: {prediction[0]}")


## Running the main function
if __name__ == '__main__':
    main()

