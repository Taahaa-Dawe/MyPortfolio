import streamlit as st
import pandas as pd
from PIL import Image

option = ""

def select(option):
	DisplayPart(option)

import streamlit as st
import streamlit.components.v1 as components

def Clustering():
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
    .btn {
            color: white;
            height: 40px;
            width: 100px;
            padding: 2px;
            text-decoration: none;
        }
    #atag{
            text-decoration: none;
        }
      .nav {
            width: 50vw;
            height: 10vh;
            font-size: Large;
            text-align: center;
            margin: 10px 5px;
            display: flex;
            flex-direction: row;
            justify-content: space-around;
            place-items: center;
        }
</style>
    """,
    unsafe_allow_html=True
)
	st.header("Clustering", divider = "blue")

	header = st.container()

	header.write(
"""
<div class='fixed-header'/>
<center>
    <div class="nav"/>
	<a href="#Overview" class="btn",id ="atag">Overview</a>
        <a href="#DataPrep" class="btn",id ="atag">Data Prep</a>
        <a href="#Code" class="btn",id ="agta">Code</a>
        <a href="#Result" class="btn",id ="atag">Results</a>
	<a href="#Conclusion" class="btn",id ="atag">Conclusions</a>

</center>
""", 
	    unsafe_allow_html=True)











	st.markdown("""
<div id="Overview">
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Overview")
	st.write("""
	<p style = "line-height: 2">
		
Clustering is an unsupervised technique in machine learning designed to group data points that share similarities. These similarities are evaluated using distance or similarity metrics. 
	<br><br>
A distance metric is a measurement that adheres to specific conditions:</p> 

<ul>
	<li>It always produces a value greater than zero when comparing two distinct points, yields the same result regardless of the order of comparison
		<ul>
			<li>dis(A, B) &gt; 0</li>
			<li>dis(A, B) = dis(B, A)</li>
		</ul>
	</li>
	<li>It returns zero only when comparing identical points
		<ul>
			<li>dis(A, B) = 0 iff A = B</li>
		</ul>
	</li>
	<li>It satisfies the triangle inequality property.
		<ul>
			<li>dis(A, B) + dis(B, C) &ge; dis(A, C)</li>
		</ul>
	</li>
</ul>
	<b>  The distance metrics used for this project: </b>
<ul>
    <li>Euclidean Distance, also known as the L2-Norm, represents the geometric distance between two points within a vector space. It is a fundamental concept in mathematics and is a specific case of the Minkowski distance with <em>p = 2</em>.</li>
    	<center> <a href="https://imgbb.com/"><img src="https://i.ibb.co/rmyvH50/image.png" alt="image" border="0"></a> </center>
	<br>
    <li>Cosine similarity refers to a mathematical measure used to determine the similarity between two vectors in a multi-dimensional space. It calculates the cosine of the angle between these vectors. 

When two vectors are highly similar, they will have a small angle between them, resulting in a cosine similarity close to 1. Conversely, if the vectors are orthogonal (perpendicular) to each other, indicating no similarity, the cosine similarity will be 0. If the vectors point in opposite directions, the cosine similarity will be negative, indicating dissimilarity.


In essence, cosine similarity quantifies the similarity in direction between two vectors, with values ranging from -1 (completely dissimilar) to 1 (completely similar), with 0 indicating no similarity.</li>
	<br>
	<center> <a href="https://ibb.co/2gM6J7L"><img src="https://i.ibb.co/xJgsZSc/image.png" alt="image" border="0"></a> </center>
	<center><p><a href ="https://weaviate.io/blog/distance-metrics-in-vector-search">weaviate</a></center>
</ul>


	""", unsafe_allow_html=True)
	st.write("""
	Data clustering can be categorized into three main types: Partitional clustering, Hierarchical clustering, and Density-based clustering.
	<br><br>
	<b> Partitional clustering:</b>
		<p>In partitional clustering, we need to decide how many clusters we want beforehand. The aim is to group data in a way that minimizes the distance within each group while maximizing the distance between groups. However, the initial clusters we choose can affect the final result, sometimes causing inconsistent groupings. A commonly used method for this type of clustering is called k-means. </p>
			<br>	
			<center> <a href="https://ibb.co/55BjJs5"><img src="https://i.ibb.co/xMJgrzM/image.png" alt="image" border="0"></a></center>
			<center><a herf ="https://computing4all.com/courses/introductory-data-science/lessons/a-few-types-of-clustering-algorithms/">computing4all</a></center><br>
	<b> Hierarchical clustering:</b><br><br>
		<p>
    Unlike partitional clustering, this method does not require us to set the number of clusters beforehand. It operates in either a bottom-up (agglomerative) or top-down (divisive) manner. In the bottom-up approach, each data point starts as its own cluster and then gets merged based on similarity. In the top-down approach, the entire dataset begins as one cluster and then splits into smaller clusters. Various techniques are used to calculate similarities between clusters, such as single linkage, complete linkage, and ward linkage. Common algorithms for these approaches include AGNES for agglomerative clustering and DIANA for divisive clustering.
		</p>
		<br>
		<center><a href="https://ibb.co/qd8B4Qm"><img src="https://i.ibb.co/CzyBXGV/image.png" alt="image" border="0"></a> </center>
		<center> <a href ="https://harshsharma1091996.medium.com/hierarchical-clustering-996745fe656b"> Medium</a></center><br>
	<b> Density clustering: </b> <br><br>
	<p>

Density clustering, unlike partitional clustering, doesn't need the number of clusters to be determined beforehand. It operates by grouping data based on density, making it particularly effective when data doesn't naturally form distinct clusters. The popular algorithm used for density clustering is DBSCAN.
	</p><br>
	<center><a href="https://ibb.co/m9gqLM2"><img src="https://i.ibb.co/X7MshvK/image.png" alt="image" border="0"></a></center>
	<center><a href ="https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/"> Analytics Vidhya</a><center><br>

""",unsafe_allow_html=True)
	st.write("""  		In clustering, we can explore whether we can group together the control and treatment groups from A/B testing and identify any patterns or similarities between them. This analysis can help us understand if there are any common characteristics or trends shared between the two groups, potentially providing insights into the effectiveness of the treatment compared to the control.
""")















	st.markdown(""" 
<div id="DataPrep"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Data Prep")
	st.write("""
	<br><br>



	""",unsafe_allow_html=True)








	st.markdown(""" 
<div id="Code"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Code")
	st.write("""

	""",unsafe_allow_html=True)

	st.markdown(""" 
<div id="Result"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Results")
	st.write("""
	

	""",unsafe_allow_html=True)
	st.markdown(""" 
<div id="Conclusion"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)

	st.subheader("Conclusion")
	st.write("""


	""",unsafe_allow_html=True)






def ARM():
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
    .btn {
            color: white;
            height: 40px;
            width: 100px;
            padding: 2px;
            text-decoration: none;
        }
    #atag{
            text-decoration: none;
        }
      .nav {
            width: 50vw;
            height: 10vh;
            font-size: Large;
            text-align: center;
            margin: 10px 5px;
            display: flex;
            flex-direction: row;
            justify-content: space-around;
            place-items: center;
        }
</style>
    """,
    unsafe_allow_html=True
)
	st.header("Association Rule Mining", divider = "blue")

	header = st.container()

	header.write(
"""
<div class='fixed-header'/>
<center>
    <div class="nav"/>
	<a href="#Overview" class="btn",id ="atag">Overview</a>
        <a href="#DataPrep" class="btn",id ="atag">Data Prep</a>
        <a href="#Code" class="btn",id ="agta">Code</a>
        <a href="#Result" class="btn",id ="atag">Results</a>
	<a href="#Conclusion" class="btn",id ="atag">Conclusions</a>

</center>
""", 
	    unsafe_allow_html=True)











	st.markdown("""
<div id="Overview">
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Overview")
	st.write("""
Association rule mining, a form of unsupervised learning, delves into the depths of data without predefined labels or outcomes. It autonomously identifies hidden patterns and associations among variables in large datasets, such as those in market basket analysis. This approach liberates businesses from the need for labeled data, allowing for the exploration of intricate relationships between items and revealing valuable insights into customer behavior.
<center> 
<a href="https://ibb.co/FHCzybY"><img src="https://i.ibb.co/DrTQvDM/image.png" alt="image" border="0"></a>
</center> 

<center>
  <a href="https://www.datacamp.com/tutorial/market-basket-analysis-r">Data Camp</a>
</center>
<br><br>
The example above mentioned is a classic illustration often used to explain association rule mining (ARM). It's an example where analysis of transaction data reveals correlations between seemingly unrelated items, such as bread, beer, milk, eggs, and diapers. This example highlights how association rule mining can uncover valuable insights, like the tendency of customers to purchase certain items together, facilitating targeted marketing strategies and product placement optimizations.
<br><br>
<p><b>Association rule mining relies on three key parameters:</b></p>
<li><b>Support</b>: This parameter measures the frequency of occurrence of a particular itemset in the dataset. It signifies how often an itemset appears in all transactions. High support indicates that the itemset is frequently bought together.
<br><br>
<center>
	<a href="https://imgbb.com/"><img src="https://i.ibb.co/MPwPzmn/image.png" alt="image" border="0"></a>
 </center><br>


<li><b>Confidence</b>: Confidence measures the reliability of the association rule. It indicates the likelihood that an item B is purchased when item A is purchased, expressed as the ratio of the number of transactions where both A and B are bought to the number of transactions where A is bought.
<br><br>
<center> 
	<a href="https://imgbb.com/"><img src="https://i.ibb.co/KsQjtfy/image.png" alt="image" border="0"></a>
</center><br>
<li><b>Lift</b>: Lift assesses the strength of the association between two items. It compares the likelihood of both items being bought together to the likelihood of their independent occurrence. A lift greater than 1 indicates that the items are positively correlated, meaning their occurrence together is more likely than random chance. A lift of 1 indicates independence, while a lift less than 1 suggests a negative correlation.
<br><br>
<center> 
	<a href="https://imgbb.com/"><img src="https://i.ibb.co/NS2LVdF/image.png" alt="image" border="0"></a>
</center><br>

<b>The need for optimization and Apriori:</b>
<br>
In a dataset with n elements in transactions, there are 2^n subsets, leading to an immense number of possible association rules. To manage this large rule space, programming languages utilize the Apriori algorithm, which employs pruning. If a rule like A -> B doesn't meet the minimum support requirement, Apriori avoids exploring any larger sets containing A -> B. This approach reduces computational workload by focusing only on promising rule combinations, thus improving the efficiency and scalability of the algorithm.
<br><br>
<center><a href="https://ibb.co/X8WCsdz"><img src="https://i.ibb.co/q5Drs4R/image.png" alt="image" border="0"></a></center>
<center><a href ="https://gatesboltonanalytics.com/">Amy Gates</a></center>	<br>

Association rule mining (ARM) is a valuable technique for analyzing data related to marketing campaigns and customer behaviors. By examining patterns and associations within the dataset, ARM can uncover insights that help marketers optimize their campaign strategies. For example, ARM might reveal that certain combinations of customer behaviors, such as low search activity and high impressions, are strongly associated with specific types of marketing campaigns, like control campaigns. Armed with this knowledge, marketers can tailor their campaigns to better target and engage customers based on their observed behaviors. Additionally, ARM can assist in segmentation and personalization efforts by identifying distinct customer segments with unique preferences and behaviors. This enables marketers to deliver more personalized messaging and offers, leading to improved campaign effectiveness. Furthermore, ARM can support A/B testing initiatives by helping marketers design experiments that test different campaign approaches and interpret the results more effectively. Overall, ARM provides marketers with valuable insights into the complex dynamics between customer behaviors and marketing campaigns, empowering them to make data-driven decisions that drive better outcomes.

""", unsafe_allow_html=True)












	st.markdown(""" 
<div id="DataPrep"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Data Prep")
	st.write("""


	""",unsafe_allow_html=True)








	st.markdown(""" 
<div id="Code"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Code")
	st.write("""
	
""",  unsafe_allow_html=True)
	st.subheader("Results")
	st.write("""
	

	""",unsafe_allow_html=True)
	html_file = open("pages/ARMRules.html", 'r', encoding='UTF-8')
	source_code = html_file.read()
	html_file.close()
	components.html(source_code, height=550)
	
	html_file = open("pages/ARMRules1.html", 'r', encoding='UTF-8')
	source_code = html_file.read()
	html_file.close()
	components.html(source_code, height=550)
	
	st.markdown(""" 
<div id="Conclusion"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)

	st.subheader("Conclusion")
	st.write("""
	


""",unsafe_allow_html=True)



## Introduction 
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
	st.write("""
			<p style = "line-height: 2" >
The data on online dating trends in the United States presents a nuanced picture of the phenomenon's prevalence, demographics, and experiences. It reveals that while around three-in-ten adults have utilized dating sites or apps, there are significant variations across demographic groups. Younger individuals, LGB adults, and those with higher education levels are more likely to engage in online dating.
				<br>
			</p>
  <center> <a href="https://ibb.co/XjTjR1q"><img src="https://i.ibb.co/Dznz2hZ/image.png" alt="image" border="0"></a></center>
    <center><a href ="https://www.pewresearch.org/short-reads/2023/02/02/key-findings-about-online-dating-in-the-u-s/">Pew Research</a></center>
   	<br>
   <p style = "line-height: 2" >
	Popular platforms like Tinder, Match, and Bumble dominate the market, with differences in usage patterns among age groups and sexual orientations. Despite mixed experiences reported by users, with slightly more leaning towards positive encounters, concerns about safety persist, particularly among older adults and non-users. Support for background checks on dating profiles is widespread, indicating a desire for enhanced security within the online dating landscape.
 
   </p>

   <center> <a href="https://ibb.co/xmhVrdg"><img src="https://i.ibb.co/9wnRdS4/image.png" alt="image" border="0"></a> </center>
   <center><a href ="https://www.pewresearch.org/short-reads/2023/02/02/key-findings-about-online-dating-in-the-u-s/">Pew Research</a></center>

   <br>

 <p style = "line-height: 2" >
   Moreover, gender disparities in online dating experiences are evident, with women more likely to encounter unwanted behaviors such as receiving unsolicited sexually explicit messages or continued unwanted contact. Conversely, men are more prone to feelings of insecurity due to a lack of messages. The reasons for using online dating platforms vary, ranging from seeking long-term partners to casual dating, with men more inclined towards casual sex as a major motivator. Despite some skepticism, a substantial proportion of Americans believe online dating facilitates the search for partners, although opinions on the abundance of choices available differ. Overall, the data underscores the complex interplay of usage patterns, perceptions, and safety concerns within the realm of online dating in contemporary society.
</p>

   """	, unsafe_allow_html=True)
	

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
	elif option == "DataPrep/EDA":
		DataCleaning()
	elif option == "Clustering":
		Clustering()
	elif option == "ARM":
		ARM()
	else:
		st.write("Work in Progress")


st.title("Exploring the Landscape of Modern Romance through Text Mining and Machine Learning Algorithms")


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
