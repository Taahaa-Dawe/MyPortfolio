import streamlit as st
import pandas as pd
from PIL import Image
from PIL import Image
import streamlit.components.v1 as components

option = ""

def select(option):
	DisplayPart(option)


st.title("Amazigh to English Language Translation | Preserving and Honoring the Amazigh Identity.")
st.write("Authors and Emails:")
st.markdown("- Ghizlane Rehioui: ghizlane.rehioui@colorado.edu")
st.markdown("- Taahaa Dawe: taahaa.dawe@colorado.edu")

header = st.container()
with header:
	option = st.selectbox("", ("Introduction", "DataPrep/EDA", "Clustering", "ARM","LDA","DT","NB","SVM", "NN","Conclusions"))
header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
select(option)
### Custom CSS for the sticky header
st.markdown(
    """
<style>
    div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
        position: sticky;
        top: 2.875rem;
        z-index: 999;
	background-color: #0c1415;
    }
    .fixed-header {
        border-bottom:1px ;
    }
</style>
    """,
    unsafe_allow_html=True
)
