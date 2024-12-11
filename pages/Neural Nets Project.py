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
	st.write("""Amazigh languages belong to the Afro-Asiatic language family and are considered one of its most homogeneous branches. Historically, particularly in the French academic traditions, they have often been regarded as a single language. The Amazigh languages are spoken by approximately 14 million people, primarily in scattered communities across the Maghreb region of North Africa, stretching from Egypt to Mauretania, with the largest concentration in Morocco <a href ="https://www.britannica.com/topic/Amazigh-languages">(Britannica)</a>.""", unsafe_allow_html=True)
	st.write(""" <center>
 	<a href="https://ibb.co/GPvJbS3"><img src="https://i.ibb.co/h8RMtz1/image.png" alt="image" border="0"></a><br /><a target='_blank' href='https://www.instagram.com/amazighworldnews/p/C8SmeVtNzN3/'>Figure 1: Percentages of Amazigh Population in Africa (African Mapper).</a><br />
   	</center> """, unsafe_allow_html=True)
	st.write("""Prominent Amazigh languages include Tashelhit (also known as Tashelhiyt, Tashelhait, or Shilha), Tarifit, Kabyle, Tamazight, and Tamahaq. The family also encompasses extinct languages like the Guanche languages of the Canary Islands, Old Libyan (Numidian), and Old Mauretanian, known through inscriptions that remain insufficiently studied. Another possible member is Iberian, associated with the Iberian Peninsula’s early inhabitants. The ancient Tifinagh script, linked to early Libyan inscriptions and the Phoenician quasi-alphabet, continues to be used by the Tuareg people (Britannica).""")
	st.write("""<center><a href="https://ibb.co/QYZcFNY"><img src="https://i.ibb.co/tYShMsY/image.png" alt="image" border="0"></a><br /><a target='_blank' href='https://imgbb.com/'>Figure 2: Tamazight spelt in Tifinagh (left) and Romanized (right).</a><br /></center> """ , unsafe_allow_html=True)
	st.write("")
	st.write(""" The Amazigh tribes have deep historical roots in North and sub-Saharan Africa, going back more than 20,000 years and pre-dating the Arab conquest of the region. The word “Amazigh” means “Free Men”. They are the descendants of the indigenous tribes who inhabited the area for thousands of years. The Amazigh were otherwise called “Berbers”, a term that the Amazigh disliked as it stems from the Roman description of them as barbarians and was then applied by waves of invaders of North Africa across the centuries <a href = "https://www.insightvacations.com/blog/meet-moroccos-berbers/">(Insight Vacations)</a>.""", unsafe_allow_html=True)
	st.write("""In 2010, the Moroccan government launched Tamazight TV to promote the Tamazight language and culture. On July 29, 2011, Tamazight was recognized as an official language in the Moroccan Constitution, marking a significant step toward its preservation and integration. Despite these advancements, Tamazight remains underrepresented in public and scientific domains. The language comprises three main dialects: Tachelhit, spoken by 14% of the population, Tamazight by 7.6%, and Tarifit by 4.1% <a href ="https://www.moroccoworldnews.com/2015/10/170696/icram-questions-figures-hinting-at-decrease-of-amazigh-speakers-in-morocco">(Morocco World News).</a>""", unsafe_allow_html=True)
	
	st.write("""<center><a href="https://ibb.co/LvLK613"><img src="https://i.ibb.co/d7dZ6DR/image.png" alt="image" border="0"></a><br /><a target='_blank' href='https://imgbb.com/'>Figure 3: Linguistic Map of Morocco – 3 main languages (left) and 3 Amazigh dialects (right).</a><br /></center>""", unsafe_allow_html=True)
	st.write("")
	st.write("""Approximately 7.5 million Amazigh speakers reside in Morocco (<a href ="https://www.crossroadsculturalexchange.com/blog/about-amazigh">Crossroads Cultural Exchange</a>). According to <a href ="https://www.euronews.com/2023/01/31/morocco-amazigh">EuroNews </a>, Amazigh is spoken by 25% – 30% of the population, but its use is declining among younger generations. Although Amazigh is taught in some schools, the percentage of children learning the language has dropped from 14% in 2010 to just 9% in 2023. This decline can be attributed to the limited teaching in schools and the increasing use of Romanized Tifinagh, which overshadows its original script. """, unsafe_allow_html=True)
	st.write("""<center><a href="https://ibb.co/6PqD6B5"><img src="https://i.ibb.co/jMj37RF/image.png" alt="image" border="0"></a><br /><a target='_blank' href='https://imgbb.com/'>Figure 4: Students in a Moroccan classroom with names in Tifinagh <a href = "https://www.instagram.com/amazighworldnews/p/C7rcG3cMg7O/?img_index=1">(Amazighworldnews).</a></a><br /> </center>""", unsafe_allow_html=True)
	st.write("")
	st.write("""As a consequence, many Moroccans with Amazigh origins are unable to speak the language. Worsening the fact that the language is not largely taught in schools, most people write it in its original letters Tifinagh but rather use the romanized version, which might actually be another contributor to why it has not yet died.""", unsafe_allow_html=True)
	st.write("""The Expanded Graded Intergenerational Disruption Scale (EGIDS) is a tool used to assess the status of a language, measuring its level of endangerment or development (Ethnologue). The scale helps highlight a language’s strongest and safest areas of use while offering insights into its long-term viability. Standard Moroccan Tamazight is classified at EGIDS Level 1 (National), meaning it is used in education, work, media, and government in Morocco. In the language cloud, a large colored dot represents Tamazight’s institutional support, corresponding to EGIDS Levels 0 – 4 <a href ="https://www.ethnologue.com/faq/what-is-egids/">(Ethnologue)</a>.""", unsafe_allow_html=True)
	st.write("""<center><a href="https://imgbb.com/"><img src="https://i.ibb.co/Zmh3mDK/image.png" alt="image" border="0"></a><br /><a target='_blank' href='https://imgbb.com/'>Figure 5: Moroccan Tamazight on EGIDS (Ethnologue).</a><br /></center>""",unsafe_allow_html=True)
	st.write("")
	st.write("""With this, the project will revolve around Amazigh-to-English translation.This project aligns perfectly with a unique opportunity presented by Morocco’s shift from a French-focused education system to one that prioritizes English. As English proficiency grows among Moroccans, the government has also demonstrated a renewed commitment to the Amazigh language, increasing its budget by 50% to 300 million dirhams ($30 million) this year and pledging to hire hundreds of clerks to integrate the language into public services (EuroNews). Against this backdrop, developing an Amazigh-to-English translation system is not only timely but also essential for preserving cultural heritage while addressing modern linguistic needs.""")

	


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
