import streamlit as st
import pandas as pd
from PIL import Image
import streamlit.components.v1 as components
option = ""

def select(option):
	DisplayPart(option)

import streamlit as st
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
	<center><a href ="https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/"> Analytics Vidhya</a><center>
""",unsafe_allow_html=True)















	st.markdown(""" 
<div id="DataPrep"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Data Prep")
	st.write("""
		In clustering, we can explore whether we can group together the control and treatment groups from A/B testing and identify any patterns or similarities between them. This analysis can help us understand if there are any common characteristics or trends shared between the two groups, potentially providing insights into the effectiveness of the treatment compared to the control.
	<br><br>
		The dataset started off well-structured with labeled data. However, for our clustering analysis, we discarded the labels and focused exclusively on columns containing numerical values that had substantial relevance to our analysis. This preprocessing step was crucial because distance metrics such as Euclidean and cosine are only suitable for numeric data. Additionally, we normalized and standardized the data to ensure consistency and comparability across different attributes, thereby enhancing the accuracy of our clustering analysis.
	<center><a href="https://ibb.co/5k5CpTR"><img src="https://i.ibb.co/hcKrQsm/image.png" alt="image" border="0"></a></center>
	<center>Data Before Transforming</center><br>
	<center><a href="https://imgbb.com/"><img src="https://i.ibb.co/ZYp7th8/image.png" alt="image" border="0"></a></center>
	<center>Data After Transforming</center>	
	<a href ="https://drive.google.com/file/d/1NDDKA6fWW65en4QouqHsR6ZE1ONleatW/view?usp=sharing">Link To Sample Data </a>


	""",unsafe_allow_html=True)








	st.markdown(""" 
<div id="Code"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Code")
	st.write("""
	<ul>
	<li>Link to the Hierarchical clustering in R:  <a href = "https://github.com/Taahaa-Dawe/Machine_Learning_Project_AB_Testing/blob/main/hclust_using_r">Hclust </a>
	<li>Link to the K-means clustering in python: <a href = "https://github.com/Taahaa-Dawe/Machine_Learning_Project_AB_Testing/blob/main/kmeans.py"> K-means</a>

	""",unsafe_allow_html=True)

	st.markdown(""" 
<div id="Result"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Results")
	st.write("""
	<b>Hierarchical clustering:</b><br>
 
<center><a href="https://ibb.co/K2cF1x0"><img src="https://i.ibb.co/n7qzWMg/image.png" alt="image" border="0"></a></center>
<center>Hierarchical clustering using Euclidean distance</center>
<br><br><center><a href="https://ibb.co/b7yy1hM"><img src="https://i.ibb.co/stDDjSd/image.png" alt="image" border="0"></a></center>
<center>Hierarchical clustering using Cosine Similarity</center><br>

<p>
	The Cosine clustering method indicates two distinct clusters, whereas the Euclidean distance measure didn't work well in hierarchical clustering (hclust). There are a few exceptions where the test and control groups were classified incorrectly. 
</p>
	<b>K-means clustering:</b><br>
	<center> <a href="https://ibb.co/R2QkqjK"><img src="https://i.ibb.co/qBm3vJ8/image.png" alt="image" border="0"></a> </center> 
	<center> Distance matrix for 4 control and 4 test conditions.</center>
	<br>
	<center><a href="https://imgbb.com/"><img src="https://i.ibb.co/RcWF7QL/image.png" alt="image" border="0"></a></center>
	<br>
The above picture depicts K-means clustering with a specified number of clusters, which is 2. The features used in plotting the clusters tend to provide a good segregation of two labels.
<br><br>
<center><a href="https://ibb.co/8P9GN3V"><img src="https://i.ibb.co/h2VrL5v/image.png" alt="image" border="0"></a></center>
<center>Silhouette value for K=2</center><br>
<b>Different Values of K and their Silhouette Score</b>
<center><a href="https://ibb.co/3p3D3NT"><img src="https://i.ibb.co/QPg2gNJ/image.png" alt="image" border="0"></a></center>
<center>Silhouette value for K=3</center><br>
<center><a href="https://ibb.co/F32jc2h"><img src="https://i.ibb.co/GpybGyd/image.png" alt="image" border="0"></a></center>
<center>Silhouette value for K=4</center><br>
<center><a href="https://ibb.co/kHd2xkH"><img src="https://i.ibb.co/f1cd2B1/image.png" alt="image" border="0"></a></center>
<center>Silhouette value for K=5</center><br>

<center><a href="https://imgbb.com/"><img src="https://i.ibb.co/8nC7YDk/image.png" alt="image" border="0"></a></center> 
<center>Comparison of number of clusters and their Silhouette scores</centre> <br><br>

	""",unsafe_allow_html=True)
	st.write("""
The silhouette analysis suggests that the maximum silhouette average value occurs at 4, indicating that this could be considered the optimal number of clusters.

However, based on our prior knowledge, we understand that the data can only be classified into two groups. Therefore, we will not accept the value of <strong>k</strong> as 4.

<center><a href="https://imgbb.com/"><img src="https://i.ibb.co/5FHdSMY/image.png" alt="image" border="0"></a></center>
<center>WCSS (Elbow Method)</center>
The elbow method suggests that the optimal number of clusters is 2, which aligns with our data where we have only two classes.
	<br><br>
	<center><a href="https://imgbb.com/"><img src="https://i.ibb.co/t3R3PR3/image.png" alt="image" border="0"></a></center>
	<center><b> Confusion matrix </b></center>
	The model classified 7 True Negatives and 9 False Positives, with an overall accuracy of 70.2 percent.
	<br><br>
	<b> Comparison of Hierarchical Clustering and K-means: </b>
	<br>In the given scenario, both hierarchical clustering (hclust) and K-means clustering suggested 2 clusters, aligning with the number of labels available in the data. This suggests that both methods have successfully identified the underlying structure of the data into two distinct groups or classes. However, despite suggesting the correct number of clusters, there may be instances where individual data points are misclassified within those clusters. The overall accuracy of the model is reported as 72 percent, indicating that while a majority of the data points are correctly classified, there are still some misclassifications present.


	""",unsafe_allow_html=True)
	st.markdown(""" 
<div id="Conclusion"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)

	st.subheader("Conclusion")
	st.write("""
Clustering is essential for organizing AB testing data into clear groups, confirming variations in spending habits between control and test users. It sheds light on purchasing behaviors, providing deeper insights into user characteristics and preferences within each group. This insight helps refine marketing strategies, products, and customer experiences tailored to specific segments. By understanding behavioral nuances, businesses can optimize decision-making and improve overall performance based on a nuanced understanding of user behavior and preferences.
	""",unsafe_allow_html=True)


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
	st.write("""
<p style = "line-height: 2" >
According to an article titled <a href ='https://hbr.org/2017/06/a-refresher-on-ab-testing'>"A Refresher on A/B Testing"</a>&nbsp;published by the Harvard Business Review in June 2017, the method known as A/B testing was developed almost a century ago by Ronald Fisher. Initially utilized in agricultural and medical experiments, its adaptation to online environments in the 1990s revolutionized how businesses optimize websites, apps, and marketing strategies
</p>
"""	, unsafe_allow_html=True)
	st.write("""
<p style = "line-height: 2" >


By randomly assigning users to different versions and analyzing metrics like click-through rates, companies can swiftly assess the impact of changes and make data-driven decisions.
However, despite its widespread adoption, many organizations fall prey to common pitfalls such as premature conclusions, excessive metric tracking, and neglecting the importance of retesting.


</p>
"""	, unsafe_allow_html=True)
	
	st.write(""" <center>
<a href="https://ibb.co/nPPbby7"><img src="https://i.ibb.co/5KK663r/image.png" alt="image" border="0"></a></center>
"""	, unsafe_allow_html=True)
	st.caption("<center><a href = 'https://zapier.com/blog/ab-testing-email-marketing/'>_zapier</a></center>", unsafe_allow_html=True)
	st.write("""
<p style = "line-height: 2" >

Nonetheless, A/B testing remains invaluable for its ability to provide quick insights and facilitate iterative improvements, enabling businesses to adapt rapidly to changing consumer preferences and market dynamics. While A/B testing offers valuable insights, its effectiveness hinges on proper execution and interpretation.

As companies strive to maximize online engagement and conversion rates, understanding the nuances of experimental design and statistical analysis becomes paramount. By adopting best practices and leveraging the flexibility of digital platforms, businesses can harness the power of A/B testing to iteratively refine their strategies, enhance user experiences, and drive sustainable growth in an increasingly competitive landscape.

</p>
"""	, unsafe_allow_html=True)
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
