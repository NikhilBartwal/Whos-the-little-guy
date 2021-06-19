import os
import numpy as np
import pandas as pd

from utils import *
from inference import *
st.set_page_config(layout="wide")

def load_homepage():
    info_homepage = st.beta_container()
    uploaded_file = st.file_uploader('Please upload an image->')

    if uploaded_file:
        st.warning('Please close the image to go back to the homepage')

        #Since the file is uploaded as a Bytes object in Streamlit, we need to
        # process it to PIL image object to display
        image = process_image(uploaded_file)
        #Display the image and accept the desired category from the user
        predict_button = display_image(image)

        #The actual model loading and inference is used when the predict button is clicked
        if predict_button:
            model, input_details, output_details = load_model()
            model_output = run_inference(model, image, input_details, output_details)
            #Create a list of all 104 flower species from labels txt files
            labels = get_labels()

            #Calculate and display the top 3 predictions along with the species info
            display_inference(labels, model_output)

    else:
        with info_homepage:
            st.write("Who's The Little Guy?")
            st.write("Do you know that there are currently 40000 species of different\
             types of plants and flowers. Let's identify some")
            st.write("Le's get started by uploading an image: ")

def display_sidebar(options):
    st.sidebar.warning('Please upload the image in a standard format (jpg, jpeg, png)')
    option = st.sidebar.selectbox('Explore the following:', options)
    return option

def load_catalog():
    labels = get_labels()
    display_inference(labels, for_catalog=True, num_predictions=104)

def main():
    options = ['Homepage', 'Flower Catalog']
    option = display_sidebar(options)

    if option == options[0]:
        load_homepage()
    elif option == options[1]:
        load_catalog()

if __name__=='__main__':
    main()
