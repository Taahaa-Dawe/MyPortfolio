import streamlit as st
import pandas as pd
from PIL import Image
from PIL import Image
import streamlit.components.v1 as components

option = ""

def select(option):
	DisplayPart(option)
def Abstract():
	st.header("Abstract", divider='blue')
	st.write(""" Amazigh languages, spoken by 14 million people across North Africa, face challenges in preservation and technological integration. This project develops an Amazigh-to-English translation system using Seq2Seq, Transformer, fine-tuned Helsinki-NLP models, and Google Translate API. The fine-tuned pre-trained Transformer Helsinki-NLP model achieved the highest BLEU score (49.27), highlighting its potential for under-resourced languages, despite having only Romanized support. Challenges faced in this project include limited data and lack of Tifinagh script support. This emphasizes the need for more resources to preserve Amazigh’s linguistic identity through technology.""")

def Intro():
	st.header("Introduction", divider='blue')
	st.write("""Amazigh languages belong to the Afro-Asiatic language family and are considered one of its most homogeneous branches. Historically, particularly in the French academic traditions, they have often been regarded as a single language. The Amazigh languages are spoken by approximately 14 million people, primarily in scattered communities across the Maghreb region of North Africa, stretching from Egypt to Mauretania, with the largest concentration in Morocco <a href ="https://www.britannica.com/topic/Amazigh-languages">(Britannica). </a>""", unsafe_allow_html=True)
def DataPreparation():
	pass 
def Methodology():
	pass
def AnalysisResults():
	pass
def Conclusions():
	pass
def DisplayPart(option):
	if option == "Abstract":
		Abstract()
	elif option == "Introduction":
		Intro()
	elif option == "Data Preparation":
		DataPreparation()
	elif option == "Methodology":
		Methodology()
	elif option == "Analysis and Results":
		AnalysisResults()
	elif option =="Conclusions":
		Conclusions()
	else:
		st.write("Work in Progress")

st.title("Amazigh to English Language Translation | Preserving and Honoring the Amazigh Identity.")


header = st.container()
with header:
	option = st.selectbox("", ("Abstract","Introduction", "Data Preparation", "Methodology", "Analysis and Results","Conclusions"))
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
