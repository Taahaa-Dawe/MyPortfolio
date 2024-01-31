import streamlit as st
import pandas as pd
from PIL import Image
option = ""

def select(option):
	DisplayPart(option)
def Intro():
	st.header("Introduction")
	st.write("""
		<p style = "line-height: 2" >A/B testing, also known as split testing, has emerged as a fundamental methodology in modern marketing, driving a shift towards data-centric decision-making. With more than 50% of marketers actively employing A/B testing, according to 99firms.com, its prevalence underscores its integral role in refining and optimizing digital experiences. This methodology involves comparing two versions, A and B, of a specific element, such as a webpage or email campaign, to determine which performs better based on predefined metrics. The widespread adoption of A/B testing reflects its efficacy in providing valuable insights into user behavior, preferences, and the overall effectiveness of various digital elements, contributing to a more iterative and informed approach to marketing strategy. As businesses strive for more efficient marketing strategies, the global A/B testing software market is anticipated to reach a substantial value of $1.08 billion by 2025, as reported by Business Insider in 2019. This forecast underscores the increasing importance and adoption of A/B testing on a global scale. In response to the demand for systematic optimization, 56.4% of businesses have implemented test prioritization frameworks, indicating a structured and strategic approach to refining their overall marketing strategies (99firms.com). The adoption of these frameworks highlights a commitment to ongoing improvement and efficiency in digital marketing practices.
</p>"""	, unsafe_allow_html=True)
	st.image(image_file = open('phototaahaa.jpg', "rb"),output_format="PNG")
	st.write("""
<p style = "line-height: 2" >Within the realm of A/B testing, landing pages, email campaigns, and pay-per-click (PPC) advertisements are prominent areas of focus for marketers. In 2019, invesp.com reported that 77% of marketers utilize A/B testing on their websites, with 60% focusing on landing pages, 59% on email campaigns, and 58% on PPC advertisements. Additionally, A/B testing extends its influence to critical elements such as call-to-action buttons, where 85% of marketers engage in optimization efforts (Revizzy, 2020). This versatility underscores A/B testing's applicability across various facets of digital marketing, allowing businesses to systematically refine user experiences and enhance overall performance. As the landscape of digital marketing evolves, A/B testing remains a crucial tool for businesses seeking to adapt to changing user preferences and market dynamics. With its capacity to provide empirical insights and guide iterative improvements, A/B testing stands at the forefront of a data-driven revolution, empowering marketers to make informed decisions and optimize their digital strategies for enhanced results in a dynamic and competitive environment.
</p>
"""	, unsafe_allow_html=True)

def DataCleaning():
	st.header("Data Cleaning")


def DisplayPart(option):
	if option == "Introduction":
		Intro()
	elif option == "Data Cleaning":
		DataCleaning()


st.title("Data-Driven Optimization: A/B Testing and Machine Learning Integration for Enhanced Digital Strategies")


header = st.container()
with header:
	option = st.selectbox("", ("Introduction", "Data Cleaning"))
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
