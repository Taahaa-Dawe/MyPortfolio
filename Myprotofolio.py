import streamlit as st
import streamlit.components.v1 as components
import base64


html_file = open("tp.html", 'r', encoding='UTF-8')
source_code = html_file.read()
html_file.close()

# Read the image file
image_file = open('phototaahaa.jpg', "rb")
contents = image_file.read()
data_url = base64.b64encode(contents).decode("utf-8")
image_file.close()


source_code_with_image = source_code.replace('{data_url}', data_url)

components.html(source_code_with_image, height=1600, scrolling=True)
