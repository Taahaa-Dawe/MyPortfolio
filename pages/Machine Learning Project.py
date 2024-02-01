import streamlit as st
import pandas as pd
from PIL import Image
option = ""

def select(option):
	DisplayPart(option)
def Intro():
	st.header("Introduction")
	st.write("""
		<p style = "line-height: 2" >In their pursuit of more efficient marketing strategies, businesses are driving the global A/B testing software market towards a projected value of $1.08 billion by 2025, according to a 2019 report by Business Insider. This anticipated growth underscores the increasing significance and widespread adoption of A/B testing as a fundamental methodology. A/B testing is at the forefront of a paradigm shift in modern marketing, steering decision-making towards a more data-centric approach. The substantial market value reflects the recognition of A/B testing as a crucial tool for optimizing marketing efforts and maximizing returns on investment. This forecast signals a transformative trend as businesses increasingly prioritize data-driven insights in shaping their marketing strategies.
</p>"""	, unsafe_allow_html=True)
	st.write(""" <a href="https://ibb.co/tZzqGzQ"><img src="https://i.ibb.co/LQrxFrY/1-A2ag-SPKf-LY9-J-w-8e3y-Ld-Q.webp" alt="1-A2ag-SPKf-LY9-J-w-8e3y-Ld-Q" border="0"></a>""" , unsafe_allow_html=True)
	st.caption("<center><a href = 'https://netflixtechblog.com/what-is-an-a-b-test-b08cc1b57962'>Netflix TechBlog</a></center>", unsafe_allow_html=True)
	st.write("""
<p style = "line-height: 2" >A/B testing involves comparing two variations (A and B) of an element and analyzing their performance to unlock invaluable insights into user behavior and preferences. This data-driven approach is instrumental in fueling smarter marketing strategies, facilitating iterative improvements, and ultimately driving success. The projected market value of $1.08 billion signifies the global acceptance of A/B testing, with a substantial portion of businesses, 56.4% according to 99firms.com, prioritizing the implementation of test frameworks.

A/B testing involves comparing two variations (A and B) of an element and analyzing their performance to unlock invaluable insights into user behavior and preferences. This data-driven approach is instrumental in fueling smarter marketing strategies, facilitating iterative improvements, and ultimately driving success.  A/B testing's versatility is evident across various marketing elements, from landing pages and email campaigns to PPC ads and call-to-action buttons. As reported by invesp.com in 2019, a staggering 77% of marketers engage in testing their websites, with a specific focus on optimizing landing pages, email campaigns, and PPC ads. The projected market value of $1.08 billion signifies the global acceptance of A/B testing, with a substantial portion of businesses, 56.4% according to 99firms.com, prioritizing the implementation of test frameworks.


Even seemingly minor elements like call-to-action buttons witness optimization efforts, with an impressive 85% of marketers actively involved, according to Revizzy in 2020. This widespread adoption underscores the immense value that A/B testing offers to marketers. In the ever-evolving digital landscape, A/B testing remains a crucial linchpin, empowering marketers with empirical data. This data guides them towards iterative improvements, spearheading a data-driven revolution in marketing strategies.

This newfound knowledge enables marketers to make strategic decisions and fine-tune their digital strategies, ensuring success in a highly competitive and dynamic environment. As the marketing landscape continues to evolve, A/B testing's role as a data-driven powerhouse is assured, propelling businesses towards unparalleled success. The anticipated substantial growth in the global A/B testing software market, as reported by Business Insider, underscores the industry's recognition of A/B testing as an indispensable tool for achieving marketing efficiency and driving business growth.</p>"""	
	, unsafe_allow_html=True)
	st.subheader("Questions to be answered")
	lst = [
    "What is the distribution of user spending across different gender categories?",
    "How does the number of purchases vary between different user groups?",
    "Can we identify any seasonal patterns in user spending behavior?",
    "Is there a correlation between the amount spent and the user's country of origin?",
    "What is the overall distribution of users based on the device they use?",
    "Are there any outliers or unusual spending patterns in the dataset?",
    "How does the average spending differ between male and female users?",
    "Is there a relationship between the user's group membership and the number of purchases?",
    "What insights can be gained from visualizing the temporal trends in user engagement metrics?",
    "Can we build a predictive model to estimate the amount spent by users based on their demographic and behavioral features?",
    "How well can we classify users into different groups using their spending patterns and demographic information?",
    "Is it possible to predict the likelihood of a user making a purchase based on their historical behavior and characteristics?",
    "Can we identify the most important features that contribute to predicting the gender of a user in the dataset?",
    "How effective is a clustering algorithm in grouping users based on their spending behavior and demographics?"
	]

	s = ''

	for i in lst:
		s += "- " + i + "\n"

	st.markdown(s)

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
