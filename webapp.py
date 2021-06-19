import os
import numpy as np
import pandas as pd

def load_homepage():
    info_homepage = st.beta_container()
    uploaded_file = st.file_uploader('Please upload an image->')

    if uploaded_file:
        st.warning('Please close the image to go back to the homepage')
        
    else:
        st.write("Who's The Little Guy?")
        st.write("Do you know that there are currently 40000 species of different\
         types of plants and flowers. Let's identify some")
        st.write("Le's get started by uploading an image: ")

def main():
    options = ['Homepage', 'Flower Catalog']
    option = display_sidebar(options)

    if option == options[0]:
        load_homepage()
    elif option == options[1]:
        load_catalog()

if __name__=='__main__':
    main()
