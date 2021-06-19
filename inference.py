import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from utils import *
# """Script used for loading the model, running and displaying inference."""

# st.cache is an amazing feature by Streamlit which caches the function output
# for a specific input and re-uses the same output when the same inputs are used
# in future. This is supported among multiple users as well.
@st.cache
def get_labels():
    """Returns list of 104 flower labels from the txt file."""

    label_path = 'labels/flower_labels.txt'
    file = open(label_path).readlines()
    labels = [data.split('\n')[0] for data in file]
    return labels

@st.cache
def load_model():
    """Loads and caches the TFLite model for inference."""

    tflite_model_path = 'model/flower_model.tflite'
    interpreter = tf.lite.Interpreter(tflite_model_path)
    #Allocate memory for the input and output tensors used by the TFLite model
    interpreter.allocate_tensors()
    # the input details contain the concerned datatype and the expected input size
    # This will come in handy while pre-processing the uploaded image
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details

@st.cache
def run_inference(model, image, input_details, output_details):
    """Runs inference on the input image and returns the probability list (unsorted)."""

    #Resizing the input image as per the model requirements
    expected_dims = input_details[0]['shape'][1:3]
    image = image.resize(expected_dims, Image.ANTIALIAS)

    #Taking only 3 channels in case of any png image that have 4 channels
    image = np.asarray(image, dtype=np.float32)[:,:,:3]/255.0
    #Since, the model expects the image in a batch, so we add another dimension
    # To make the new dimension as 1 x width x height x 3
    image = image.reshape([1, image.shape[0], image.shape[1], 3])
    #Input the image to the TFLite model and run inference with invoke()
    model.set_tensor(input_details[0]['index'], image)
    model.invoke()
    #Collect the output probabilities for all labels
    species = model.get_tensor(output_details[0]['index'])
    return species

def display_inference(model_output, labels, image):
    model_output = model_output.flatten()
    # Calculate the top 3 indices with the highest probabilities
    top_indices = model_output.argsort()[-3:][::-1]
    top_scores = [round(model_output[index]*100, 2) for index in top_indices]
    # Retrieve name of the species from the labels using indices
    top_preds = [labels[index] for index in top_indices]
    catalog_image_path = 'catalog/'

    st.write(" ----- ")
    #DIsplay catalog image with the basic and detailed info from database

    for num_prediction in range(3):
        #Divide into sub-parts for catalog image and info
        pred_image, pred_info = st.beta_columns([1,2])
        info_dict = get_image_info(top_preds[num_prediction])
        info_dict['score'] = top_scores[num_prediction]

        #Populate the image and info parts with the data retrieved from database
        with pred_image:
            st.image(catalog_image_path + str(top_indices[num_prediction]) + '.jpeg')
        with pred_info:
            display_image_info(info_dict)
        #Add additional description in ReadMore
        with st.beta_expander('Read more...'):
            st.write('\n'.join(info_dict['description'].split('.')))
