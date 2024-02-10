import streamlit as st
import pandas as pd
from PIL import Image

option = ""

def select(option):
	DisplayPart(option)
def Intro():
	st.header("Introduction", divider='blue')
	st.write("""
		<p style = "line-height: 2" >
			The advent of online dating platforms has significantly shaped the landscape of modern romance. According to a comprehensive study by Stanford sociologist Michael Rosenfeld, heterosexual couples are increasingly more likely to initiate their relationships through online channels than traditional avenues like family connections and neighborhood encounters. The study, based on a 2017 survey by <a href ="https://news.stanford.edu/2019/08/21/online-dating-popular-way-u-s-couples-meet/">Stanford News Service</a> indicates a noteworthy shift in dating dynamics, with approximately 39% of heterosexual couples reporting that they met online, a substantial increase from the 22% recorded in 2009. The rise of the graphical World Wide Web in 1995 and the subsequent proliferation of smartphones in the 2010s have played pivotal roles in transforming the dating landscape.
			<br><br>
While online dating has become a ubiquitous part of contemporary courtship, public perceptions about its impact remain diverse. The study highlights demographic variations in views, with educational backgrounds playing a significant role. Those with at least a bachelor's degree tend to perceive online dating more positively, emphasizing its capacity to expand dating options and facilitate connections with like-minded individuals. However, concerns persist regarding the safety of online dating, with 46% of Americans expressing reservations about meeting someone through these platforms as reported by <a href="(https://www.pewresearch.org/internet/2020/02/06/americans-opinions-about-the-online-dating-environment/">Pew Research Center</a>, underscoring the nuanced attitudes that coexist within the broader population.
			<br><br>
		</p>
	"""	, unsafe_allow_html=True)
	st.write(""" <center>
	<a href="https://imgbb.com/"><img src="https://i.ibb.co/LvmHsDN/Img-1.png" alt="Img-1" border="0"></a> </center>
""" , unsafe_allow_html=True)
	st.caption("<center>Views on Dating Apps</center>", unsafe_allow_html=True)
	st.write("""
			<p style = "line-height: 2" >
				<br>
				The positive aspects of online dating, as outlined by respondents, include the ability to evaluate potential partners before meeting them in person and the convenience of connecting with people who share common interests. Success stories of couples who met online are often cited as evidence of the platform's efficacy in fostering meaningful relationships. Moreover, the efficiency and accessibility of online dating are acknowledged as valuable features, allowing users to navigate the dating landscape in a more targeted and tailored manner. These perspectives highlight the multifaceted nature of online dating experiences and the factors that contribute to shaping individual opinions.
				<br><br>
			</p>"""	, unsafe_allow_html=True)
	st.write(""" <center>
	<a href="https://imgbb.com/"><img src="https://i.ibb.co/Jm9LSh9/img-2.png" alt="img-2" border="0"></a></center>
""" , unsafe_allow_html=True)
	st.caption("<center>Thoughts on how easy it is to find love</center>", unsafe_allow_html=True)
	st.write("""
			<p style = "line-height: 2" >
				<br>
				Despite the positive aspects, challenges and criticisms are also associated with online dating. Issues such as dishonesty, perceived impersonality in courtship, and concerns about safety are raised by those who hold a more negative view. Some argue that the traditional ways of meeting people, such as through friends or in-person introductions, offer a more individualized and authentic approach to building connections. The study's findings reveal the complexities surrounding the impact of online dating on dating culture, emphasizing the need for a nuanced understanding that considers the diversity of experiences and perspectives within the broader population.
				<br>
			</p>"""	, unsafe_allow_html=True)
	st.subheader("Questions to be answered")
	lst = [
    "Key themes or topics discussed in the text regarding online dating",
    "Significant trends or changes in attitudes towards online dating over time",
    "Most frequently mentioned concerns or criticisms associated with online dating platforms",
    "Demographic factors correlating with differing perceptions of online dating",
    "Comparison of positive aspects (expanded dating options, convenience) to negative aspects of online dating",
    "Specific examples or anecdotes illustrating the impact of online dating on modern romance",
    "Main features or functionalities of online dating platforms highlighted in the text",
    "Patterns in language used to describe successful versus unsuccessful online dating experiences",
    "Concerns or criticisms regarding the safety and security of online dating",
    "Alignment or divergence of perspectives in the text with broader societal attitudes or research findings on online dating"
]
	s = ''

	for i in lst:
		s += "- " + i + "\n"

	st.markdown(s)


def DataCleaning():
	st.header("Data Gathering and Cleaning", divider='blue')
	st.subheader("Data Gathering:")
	st.write("""
		<p style = "line-height: 2">
			Data gathering involved collecting information from multiple sources for comprehensive coverage and analysis. Utilizing the NEWS API and Reddit API provided real-time news updates and diverse perspectives. Additionally, data scraping from Buffz feed website enriched the dataset with niche content. Integrating data from these sources enabled a holistic approach to understanding, facilitating informed decision-making and analysis.
		</p>
"""	, unsafe_allow_html=True)

	st.write(	
""" 			<center><br>
<a href="https://ibb.co/HqCJ9Rk"><img src="https://i.ibb.co/0rKwz1d/image.png" alt="image" border="0"></a>
			</center>
"""	,unsafe_allow_html=True)
	st.caption("<center> Snapshot of Raw Data from NewsApi</center>", unsafe_allow_html=True)
	
	st.write(	
""" 			<center> <br>
<a href="https://ibb.co/ZW5pMDL"><img src="https://i.ibb.co/cD0Srng/image.png" alt="image" border="0"></a> 
			</center>
"""	,unsafe_allow_html=True)
	st.caption("<center> Snapshot of Raw Data from BuzzFeed.</center>", unsafe_allow_html=True)

	st.write(	
""" <center><br>
<a href="https://ibb.co/yyDKQkr"><img src="https://i.ibb.co/xjyw7gN/image.png" alt="image" border="0"></a> 
			</center>
"""	,unsafe_allow_html=True)

	st.caption("<center> Cleaned Data </center>", unsafe_allow_html=True)	
	st.subheader("Data Cleaning:")

	st.write("<b>Count Vectorizer</b>", unsafe_allow_html=True)
	st.write("CountVectorizer is a text preprocessing technique commonly used in natural language processing (NLP) tasks for converting a collection of text documents into a numerical representation.")
	st.write(	
""" <center><br>
<a href="https://ibb.co/yyDKQkr"><img src="https://i.ibb.co/xjyw7gN/image.png" alt="image" border="0"></a> 
			</center>
"""	,unsafe_allow_html=True)
	st.caption("<center>Data Frame of CountVectorizer</center>", unsafe_allow_html=True)
	st.write("""<center><br>
<a href="https://imgbb.com/"><img src="https://i.ibb.co/TgDws3y/image.png" alt="image" border="0"></a>
			</center>

 """,unsafe_allow_html=True)
	st.caption("<center>Word Cloud</center>", unsafe_allow_html=True)

	## Lemmatization
	st.write("<b>Lemmatization</b>", unsafe_allow_html=True)
	st.write(""" Lemmatization is the process of grouping together different inflected forms of the same word.""")
	st.write(	
""" <center><br>
<a href="https://ibb.co/3FsMF4h"><img src="https://i.ibb.co/ZgJYgGT/image.png" alt="image" border="0"></a>
	</center>
"""	,unsafe_allow_html=True)
	st.caption("<center>Data Frame of Lemmatization</center>", unsafe_allow_html=True)
	st.write("""<center><br>
<a href="https://imgbb.com/"><img src="https://i.ibb.co/kSPRmpk/image.png" alt="image" border="0"></a>
		</center>

 """,unsafe_allow_html=True)
	st.caption("<center>Word Cloud</center>", unsafe_allow_html=True)


	## Tfidf
	st.write("<b>Term Frequency-Inverse Document Frequency</b>", unsafe_allow_html=True)
	st.write("""TF-IDF stands for term frequency-inverse document frequency and it is a measure, used in the fields of information retrieval (IR) and machine learning, that can quantify the importance or relevance of string representations (words, phrases, lemmas, etc)  in a document amongst a collection of documents (also known as a corpus).""")
	st.write(	
""" <center><br>
<a href="https://ibb.co/xXj9DVR"><img src="https://i.ibb.co/QpNSfs1/image.png" alt="image" border="0"></a>			</center>
"""	,unsafe_allow_html=True)
	st.caption("<center>Data Frame of Stemming</center>", unsafe_allow_html=True)
	st.write("""
		<center><br>
<a href="https://imgbb.com/"><img src="https://i.ibb.co/CBtRfgV/image.png" alt="image" border="0"></a>
		</center>
		""",
		unsafe_allow_html=True)
	st.caption("<center>Word Cloud</center>", unsafe_allow_html=True)


	## Stemming
	st.write("<b>Stemming</b>", unsafe_allow_html=True)
	st.write("""Stemming is a text preprocessing technique in natural language processing (NLP). It reduces the inflected form of a word to its root form, also known as a "stem" or "lemma". """)
	st.write(	
""" <center><br>
<a href="https://ibb.co/GcW3D9Y"><img src="https://i.ibb.co/ZWB1rS4/image.png" alt="image" border="0"></a>
			</center>
"""	,unsafe_allow_html=True)
	st.caption("<center>Data Frame of Stemming</center>", unsafe_allow_html=True)
	st.write("""
		<center><br>
<a href="https://imgbb.com/"><img src="https://i.ibb.co/ww89Ctn/image.png" alt="image" border="0"></a>
		</center>
		""",
		unsafe_allow_html=True)
	st.caption("<center>Word Cloud</center>", unsafe_allow_html=True)

	st.subheader("Data and Code:")
	st.write("<a href = 'https://docs.google.com/spreadsheets/d/1L9qHHJK5jvXcZJbSTVTBJp553YETuQfeBbuRBsVRs1U/edit?usp=sharing'>Data</a>", unsafe_allow_html=True)
	st.write("<a href = 'https://docs.google.com/document/d/1HRjFws5aJQJA_leQlAZfkUmM_xDHAZOo3ZnJWs6-ztw/edit?usp=sharing'>Code</a>", unsafe_allow_html=True)
	
	st.subheader("API:")
	st.write("<a href = 'https://newsapi.org/v2/everything'> NewsAPI</a>", unsafe_allow_html=True)
	st.write("The other API used is from Reddit, It uses Praw Library")


def DisplayPart(option):
	if option == "Introduction":
		Intro()
	elif option == "Data":
		DataCleaning()


st.title("Exploring the Landscape of Modern Romance through Text Mining and Machine Learning Algorithms")


header = st.container()
with header:
	option = st.selectbox("", ("Introduction", "Data"))
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
