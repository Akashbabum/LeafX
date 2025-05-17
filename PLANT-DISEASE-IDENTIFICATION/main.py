import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("LeafX")
app_mode = st.sidebar.selectbox("Select Page",["HOME","DISEASE RECOGNITION"])
#app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

# import Image from pillow to open images

# Update the default image path to a reliable image URL
default_image_path = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Apple___healthy/0a5e9323-dbad-432d-ac58-d291718345d9___RS_HL%207701.JPG"

try:
    # First try to open local image
    img = Image.open("Diseases.png")
except FileNotFoundError:
    try:
        # If local image not found, load from URL
        response = requests.get(default_image_path, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes
        img = Image.open(BytesIO(response.content))
        st.image(img, caption="Welcome to LeafX", use_column_width=True)
    except requests.RequestException as e:
        st.error(f"Failed to fetch image from URL: {str(e)}")
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        # Fallback to text if image fails
        st.write("Welcome to LeafX Plant Disease Detection System")
else:
    st.image(img, use_column_width=True)

#Main Page
if(app_mode=="HOME"):
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION", unsafe_allow_html=True)
    
#Prediction Page
elif(app_mode=="DISEASE RECOGNITION"):
    st.header("DISEASE RECOGNITION")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
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