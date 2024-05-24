import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

def crop_rgb(img_path):
    img = plt.imread(img_path)
    top, bottom, left, right = 0, img.shape[0], 0, img.shape[1]
    centre = img.shape[0] // 2, img.shape[1] // 2

    left_shift, right_shift = 0, 0
    for i in range(centre[1], -1, -1):
        if np.sum(img[centre[0], i, :]) < 10:
            left_shift = centre[1] - i
            break
    for i in range(centre[1], img.shape[1]):
        if np.sum(img[centre[0], i, :]) < 10:
            right_shift = i - centre[1]
            break
    shift = max(left_shift, right_shift)
    left = centre[1] - shift
    right = centre[1] + shift
    width = right - left
    new_height = int(width * 0.60)
    centre = img.shape[0] // 2
    top = max(0, centre - (new_height // 2))
    bottom = min(img.shape[0], centre + (new_height // 2))
    cropped_img = img[top:bottom, left:right, :]
    return cropped_img

def load_and_preprocess_image(image_path, image_size=(120, 200)):
    data = pd.DataFrame({'File Path': [image_path], 'Final Label': ['RG']})
    img = crop_rgb(data['File Path'][0])
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('images/temp.jpg', gray_img)
    data = pd.DataFrame({'File Path': ['images/temp.jpg'], 'Final Label': ['NRG']})
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_dataframe(
        data, x_col='File Path', y_col='Final Label',
        target_size=image_size, class_mode='raw', batch_size=1, shuffle=False)
    return next(generator)[0]

def predict_image(model_path, image_path):
    model = load_model(model_path)
    try:
        preprocessed_image = load_and_preprocess_image(image_path)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return -1
    prediction = model.predict(preprocessed_image)
    likelihood = prediction[0][0]
    return likelihood



from menu import menu
menu()

# def app():
st.title('Glaucoma Detection')
st.write('Upload a Color Fundus Photograph (CFP) to predict the likelihood of Glaucoma.')

model_path = 'task1.h5'  # Update with the correct path to your model

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("images/uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image("images/uploaded_image.jpg", width=300)
    st.write("")
    likelihood = predict_image(model_path, "images/uploaded_image.jpg")
    if likelihood != -1:
        st.write(f"The likelihood of the image being positive for Glaucoma is {likelihood:.4f}")
        if likelihood < 0.271:
            st.success("The Eye is not Glaucomatous")
        else:
            st.error("The Eye is Glaucomatous")