from PIL import Image
import numpy as np
import streamlit as st
import sqlite3 as sql
import tempfile

@st.cache(show_spinner=True)
def process_image(uploaded_file):
    """Read uploaded file and convert it to PIL image"""
    temp = tempfile.NamedTemporaryFile(delete=True)
    temp.write(uploaded_file.read())

    image = Image.open(temp)
    return image

def display_image(image):
    """Display uploaded image along with the predict button"""
    image_column, button_column = st.beta_columns(2)

    with image_column:
        st.image(image)
    with button_column:
        st.write('Image dimensions:')
        st.write(image.size)
        predict_button = st.button('Predict!')

    return predict_button

def get_database_cursor():
    db_path = 'database/flowers.db'
    con = sql.connect(db_path)
    curr = con.cursor()
    return curr

def get_image_info(flower_name):
    curr = get_database_cursor()
    query = 'SELECT * FROM flower_data WHERE name=?'
    data = curr.execute(query, (flower_name, )).fetchall()[0]

    name, tag, description, url, taxon = data[0], data[1], data[2], data[3], data[4]
    info_dict = {
        'name': name,
        'tag': tag,
        'description': description,
        'url': url,
        'taxon': taxon
    }
    return info_dict

def display_image_info(info_dict, for_catalog=False):
    st.write('**Name:**', info_dict['name'])

    if not for_catalog:
        st.write('**Probability:**', str(info_dict['score']), '%')
    st.write('**Type:**', info_dict['tag'])

    if info_dict['taxon'] is not None:
        st.write('**Taxon:**', info_dict['taxon'])
