import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image
import numpy as np


def predict_image(image_data, model, size=(224, 224)):
        
        
        image = np.asarray(image_data)
        # Convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize image while preserving aspect ratio
        image_resized = tf.image.resize_with_pad(image, size[0], size[1], method='lanczos3', antialias=True)
        
        prediction = model.predict(image_resized)
        
        return prediction


if __name__=='__main__':

    model = tf.keras.models.load_model('~/Bootcamp/Projects/Final_Project/output/MN_model_96.h5')
    
    class_names = ['Robin','etc.'] # load from a file or write down here

    st.write("""
            # Bird Species Classifier
            """
            )

    st.write("Web app that classifies the spicies of bird!")
    file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = predict_image(image, model, size=(224,224))
        
        predicted_class = np.argmax(prediction)
        predicted_class = class_names[predicted_class]

        st.write(f"It's a {predicted_class}!")
        
        # Implement function that will show the top 5 predicted classes.

