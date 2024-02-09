import streamlit as st
import pandas as pd
from PIL import Image
import streamlit.components.v1 as components
option = ""

def select(option):
	DisplayPart(option)
def Intro():
	st.header("Introduction")
	st.write("""
		<p style = "line-height: 2" >In the pursuit of more efficient marketing strategies, businesses are driving the global A/B testing software market towards a projected value of $1.08 billion by 2025, according to a 2019 report by Business Insider. This anticipated growth underscores the increasing significance and widespread adoption of A/B testing as a fundamental methodology. A/B testing is at the forefront of a paradigm shift in modern marketing, steering decision-making towards a more data-centric approach. The substantial market value reflects the recognition of A/B testing as a crucial tool for optimizing marketing efforts and maximizing returns on investment. This forecast signals a transformative trend as businesses increasingly prioritize data-driven insights in shaping their marketing strategies.
</p>"""	, unsafe_allow_html=True)
	st.write(""" <a href="https://ibb.co/tZzqGzQ"><img src="https://i.ibb.co/LQrxFrY/1-A2ag-SPKf-LY9-J-w-8e3y-Ld-Q.webp" alt="1-A2ag-SPKf-LY9-J-w-8e3y-Ld-Q" border="0"></a>""" , unsafe_allow_html=True)
	st.caption("<center><a href = 'https://netflixtechblog.com/what-is-an-a-b-test-b08cc1b57962'>Netflix TechBlog</a></center>", unsafe_allow_html=True)
	st.write("""
<p style = "line-height: 2" >

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
	st.subheader("Data Gathering:")
	st.write("""
		<p style = "line-height: 2">The data was collected from two distinct sources to gain comprehensive insights into A/B testing scenarios. The first dataset, <a href="https://assets.datacamp.com/production/repositories/1646/datasets/2751adce60684a03d8b4132adeadab8a0b95ee56/AB_testing_exercise.csv" target="_blank">DataCamp AB Testing Exercise Dataset</a>, was obtained using the provided API. It includes 
		information on user interactions, spending patterns, and other relevant variables during A/B testing exercises. The second dataset, <a href="https://www.kaggle.com/datasets/amirmotefaker/ab-testing-dataset/code" target="_blank">Kaggle AB Testing Dataset</a>, provides insights into the impact of different campaigns on user engagement and conversions. Both datasets were meticulously cleaned to ensure accuracy and compatibility.</p>
"""	, unsafe_allow_html=True)
	st.write(	
""" 
  <center><a href="https://ibb.co/yYVVnHv" target="_blank"><img src="https://i.ibb.co/0FQQjH4/Screenshot-369.png" alt="Screenshot-369" border="0"></a></center>
"""	,unsafe_allow_html=True)

	st.caption("<center><b>Figure 1: Snapshot of Raw Data from DataCamp AB Dataset.</b></center>", unsafe_allow_html=True)	
	st.write("")
	st.write( """
  <center><a href="https://imgbb.com/" target="_blank"><img src="https://i.ibb.co/3YdhbB1/image.png" alt="image" border="0"></a></center>

""",
	 unsafe_allow_html=True)

	st.caption("<center><b>Figure 2: Snapshot of Clean Data from DataCamp AB Testing Exercise Dataset.</b></center>", unsafe_allow_html=True)
	st.write("")
	st.write( """
  <center><a href="https://ibb.co/PhbSGFw"><img src="https://i.ibb.co/d23h4jG/image.png" alt="image" border="0"></a></center>

"""	, unsafe_allow_html=True)
	st.caption("<center><b>Figure 3: Snapshot of Raw Data from Kaggle.</b></center>", unsafe_allow_html=True)

	st.write("")
	st.write( """
  <center><a href="https://ibb.co/9yDBm33"><img src="https://i.ibb.co/L9FBDhh/image.png" alt="image" border="0"></a></center>

""",
	 unsafe_allow_html=True)
	st.caption("<center><b>Figure 4: Snapshot of Cleaned Data from Kaggle.</b></center>", unsafe_allow_html=True)
	st.write()
	st.subheader("Data Cleaning:")
	st. write("""
 	<ul> DataCamp AB Testing Dataset: 
		<li> The dataset was free from missing values, NaNs, or incorrect entries.</li>
		<li> No normalization was required as the data was already in a suitable scale for analysis.</li>
  	</ul>

""" 	, unsafe_allow_html=True)
	st. write("""
 	<ul> Kaggle AB Testing Dataset:
		<li> A null value in every column of the Kaggle dataset (marked as NaN) was identified and removed, as it didn't provide valuable information.</li>
		<li> The decision to exclude this row aligns with the principle of maintaining data integrity by eliminating irrelevant or non-contributory features. </li>
  	</ul>

""" 	, unsafe_allow_html=True)
	st.write( """
  <center><a href="https://ibb.co/GT0731k"><img src="https://i.ibb.co/pyx2rsZ/image.png" alt="image" border="0"></a></center>

""",
	 unsafe_allow_html=True)
	st.subheader("Exploratory Data Analysis:")
	st.write("<b>DataCamp AB Dataset</b>", unsafe_allow_html=True)
	st.write()
	components.html("""
 <center>
 	<div class='tableauPlaceholder' id='viz1706825440189' style='position: relative'>
  		<noscript><a href='#'>
  			<img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;AB&#47;ABtestAnalysis&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a>
     		</noscript>
       		<object class='tableauViz'  style='display:none;'>
	 	<param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='ABtestAnalysis&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;AB&#47;ABtestAnalysis&#47;Dashboard1&#47;1.png' />
   		<param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' />
     		<param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                
       		<script type='text/javascript'>                    var divElement = document.getElementById('viz1706825440189');                    
	 	var vizElement = divElement.getElementsByTagName('object')[0];                    
   		if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='2227px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='2227px';} else { vizElement.style.width='100%';vizElement.style.height='3127px';}                     
     		var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    
       		vizElement.parentNode.insertBefore(scriptElement, vizElement);                
	 	</script>
</center>
 """
 	,height=2300,width=1000)

	st.write("<b>Kaggle Dataset</b>", unsafe_allow_html=True)
	st.write(
		""" <center>
<a href="https://imgbb.com/"><img src="https://i.ibb.co/dm06Db9/boxplot.png" alt="boxplot" border="0"></a> <br> </center>""",unsafe_allow_html=True)
	st.write("")
	st.write("")
	st.write("""<center>
<a href="https://ibb.co/2SSgwjN"><img src="https://i.ibb.co/HCCn0rd/corrplot.png" alt="corrplot" border="0"></a>  <br> </center> """,unsafe_allow_html=True)
	st.write("")
	st.write("")
	st.write(""" <center>
<a href="https://ibb.co/ChFwVJV"><img src="https://i.ibb.co/Tm5wPKP/newplot-1.png" alt="newplot-1" border="0"></a></center> <br>""",unsafe_allow_html=True)
	st.write("")
	st.write("")
	st.write("""<center><a href="https://ibb.co/ccDkJ4n"><img src="https://i.ibb.co/mC6bJfW/newplot.png" alt="newplot" border="0"></a></center> """,unsafe_allow_html=True) 
	st.write("")
	st.write("")
	st.write("""<center><a href="https://imgbb.com/"><img src="https://i.ibb.co/pnsG6yR/Histogram.png" alt="Histogram" border="0"></a></center>""", unsafe_allow_html=True)
	st.write("")

	st.subheader("Code:")
	st.write(""" <li> <a href="https://drive.google.com/file/d/12JXgly9Y-njQBujVTpQt3BDA55tsvPA9/view?usp=sharing" alt="Histogram" border="0">Data Cleaning of API</a> </li>""", unsafe_allow_html=True)
	st.write("""<li> <a href="https://drive.google.com/file/d/1XhJ-kGKVv2KPPQMdj1w7OVJhq6CemU6s/view?usp=sharing" alt="Histogram" border="0">Data Cleaning of Kaggle</a> </li>""", unsafe_allow_html=True)



def DisplayPart(option):
	if option == "Introduction":
		Intro()
	elif option == "DataPrep/EDA":
		DataCleaning()
	else:
		st.write("Work in Progress")


st.title("Data-Driven Optimization: A/B Testing and Machine Learning Integration for Enhanced Digital Strategies")


header = st.container()
with header:
	option = st.selectbox("", ("Introduction", "DataPrep/EDA", "Clustering", "ARM","DT","NB","SVM"))
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
