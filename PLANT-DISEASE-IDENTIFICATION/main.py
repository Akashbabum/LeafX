import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os

def model_prediction(test_image):
    try:
        # Get absolute path to model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "trained_plant_disease_model.keras")
        
        # Verify model file
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            st.write(f"Current directory contents: {os.listdir(current_dir)}")
            return None
            
        # Load model with custom object scope
        with tf.keras.utils.custom_object_scope({}):
            model = tf.keras.models.load_model(model_path, compile=False)
            st.success("Model loaded successfully")
            
        # Process image
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr]) #convert single image to batch
        
        # Make prediction
        predictions = model.predict(input_arr)
        return np.argmax(predictions)
        
    except Exception as e:
        st.error("Error in model prediction:")
        st.error(str(e))
        st.write("Debug info:")
        st.write(f"TensorFlow version: {tf.__version__}")
        st.write(f"Model file size: {os.path.getsize(model_path)} bytes")
        return None

#Sidebar
st.sidebar.title("LeafX")
app_mode = st.sidebar.selectbox("Select Page",["HOME","DISEASE RECOGNITION"])
#app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

# import Image from pillow to open images

# Update the default image path to a reliable direct image URL
default_image_path = "https://i.ibb.co/ZzV47MQP/360-F-638132571-kh-YMb1-Vwmo-XYe-KCc-Ic-VTu-TBPe-Qxnbr-TR.jpg"  # Direct link to a plant disease image

try:
    response = requests.get(default_image_path)
    img = Image.open(BytesIO(response.content))
    st.image(img, caption="Welcome to LeafX", use_container_width=True)
except Exception as e:
    st.error("Failed to load welcome image")
    st.write("Welcome to LeafX Plant Disease Detection System")

#Main Page
if(app_mode=="HOME"):
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION", unsafe_allow_html=True)
    
#Prediction Page
elif(app_mode=="DISEASE RECOGNITION"):
    st.header("DISEASE RECOGNITION")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image, use_container_width=True)
    #Predict button
    if(st.button("Predict")):
        if test_image is None:
            st.error("Please upload an image first")
        else:
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            if result_index is not None:
                #Reading Labels
                class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                              'Tomato___healthy']
                st.success("Model is Predicting it's a {}".format(class_name[result_index]))