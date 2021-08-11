import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
from src.modules.yolo3_instantiate import yolo_prediction, draw_boxes
from src.modules.visualization import image_metrics
from src.modules.preprocessing import image_cropper
import csv

def predict_images(cropped_images, model, size=(224, 224)):
        
        print_images = []
        resized_images = []

        for image in cropped_images:
            np_img = np.asarray(image)
            image_resized = tf.image.resize(np_img, size, method='lanczos3', antialias=True)
            
            print_images.append(image_resized.numpy())
            
            image_resized = np.expand_dims(image_resized, 0)
            resized_images.append(image_resized)
        
        predictions = model.predict(np.vstack(resized_images))
        
        return predictions, print_images


if __name__=='__main__':

    st.set_option('deprecation.showPyplotGlobalUse', False)

    if 'mobilenet_model' not in st.session_state:
        st.session_state.mobilenet_model = tf.keras.models.load_model('output/MN_model_96.h5')
    
    if 'yolo_model' not in st.session_state:
        st.session_state.yolo_model = tf.keras.models.load_model('output/yolo_model.h5')

    # Load labels for models
    if 'yolo_labels' not in st.session_state:
        with open('output/yolo_labels.csv', newline='') as f:
            reader = csv.reader(f)
            st.session_state.yolo_labels = [x[0] for x in reader]

    if 'class_names' not in st.session_state:
        with open('output/bird_classes.csv', newline='') as f:
            reader = csv.reader(f)
            st.session_state.class_names = [x[0] for x in reader]

    st.write("""
            # Bird Species Classifier
            """
            )

    st.write("Web app that classifies species of birds in a picture!")
    file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

    if file is None:
        st.text("Please upload an image file")
    else:

        v_boxes, v_labels, v_scores = yolo_prediction(st.session_state.yolo_model, st.session_state.yolo_labels, file)
        draw_boxes(file, v_boxes, v_labels, v_scores)
        st.pyplot()

        cropped_images = image_cropper(file, v_boxes)

        predictions, print_images = predict_images(cropped_images, st.session_state.mobilenet_model, size=(224,224))
        
        image_metrics(len(cropped_images), 1, predictions, print_images, st.session_state.class_names)
        st.pyplot()
        
       

