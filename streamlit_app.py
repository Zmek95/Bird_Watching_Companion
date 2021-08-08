import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
from src.modules.yolo3_instantiate import yolo_prediction, draw_boxes
from src.modules.visualization import image_metrics
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

    mobilenet_model = tf.keras.models.load_model('output/MN_model_96.h5')
    yolo_model = tf.keras.models.load_model('output/yolo_model.h5')

    # Load labels for models
    with open('output/yolo_labels.csv', newline='') as f:
        reader = csv.reader(f)
        yolo_labels = [x[0] for x in reader]

    with open('output/bird_classes.csv', newline='') as f:
        reader = csv.reader(f)
        class_names = [x[0] for x in reader]

    st.write("""
            # Bird Species Classifier
            """
            )

    st.write("Web app that classifies the spicies of bird!")
    file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

    if file is None:
        st.text("Please upload an image file")
    else:

        v_boxes, v_labels, v_scores = yolo_prediction(yolo_model, yolo_labels, file)
        draw_boxes(file, v_boxes, v_labels, v_scores)
        st.pyplot()
        #st.image(image, use_column_width=True)

        cropped_images = []
        for box in v_boxes:
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax

            image = Image.open(file)
            image = image.convert('RGB')

            cropped_image = image.crop((x1,y1,x2,y2))
            cropped_images.append(cropped_image)




        predictions, print_images = predict_images(cropped_images, mobilenet_model, size=(224,224))
        
        image_metrics(2, 1, predictions, print_images, class_names)
        st.pyplot()

        #st.write(f"It's a {predicted_class}!")
        
        # Implement function that will show the top 5 predicted classes.

