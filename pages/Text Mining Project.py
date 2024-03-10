import streamlit as st
import pandas as pd
from PIL import Image
from PIL import Image
import streamlit.components.v1 as components
option = ""

def select(option):
	DisplayPart(option)

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
	st.write("""  		In clustering, we aim to uncover patterns or similarities within the data by grouping similar data points together based on their features. In the context of dating data, clustering can help us identify distinct groups or clusters of individuals who share similar characteristics or behaviors in their dating experiences. By examining these clusters, we can gain insights into various aspects of dating dynamics and understand the underlying motivations or intentions of different groups of people.

For example, clustering analysis might reveal clusters of individuals who primarily use dating apps for casual hookups, while another cluster might consist of individuals seeking long-term relationships. Within each cluster, we can explore common traits such as age, interests, preferences, and communication patterns. By understanding these patterns, we can tailor dating app features or marketing strategies to better serve the needs of specific user groups.

Furthermore, clustering can help us identify outliers or anomalies in the data, such as individuals who deviate significantly from the characteristics of their respective clusters. These outliers may represent unique or unconventional dating behaviors that warrant further investigation.
""")















	st.markdown(""" 
<div id="DataPrep"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Data Prep")
	st.write("""
	<br><br>The process of transforming the text data involved utilizing the TF-IDF vectorizer, a common technique used in natural language processing to convert textual data into numerical representations. By applying TF-IDF, the importance of each word in the dataset was assessed based on its frequency in each document relative to the entire corpus. This transformation helped to capture the unique characteristics of the text data while reducing the impact of common terms that may not carry much semantic meaning.


Given that the original dataset contained a disproportionate number of rows from Reddit compared to other sources like News API and Buzzfeed, there was a noticeable imbalance in the language usage and vocabulary across the dataset. This imbalance could potentially bias the clustering algorithm towards the predominant language and jargon found in Reddit posts. To address this issue, a representative sample of 100 rows was extracted from the original dataset, ensuring a more balanced representation of language usage across different sources.

By using this sampled dataset, the clustering algorithm can better capture the underlying patterns and structure within the data, without being overly influenced by the dominant language found in a particular source. This approach enhances the reliability and generalizability of the clustering results, allowing for more robust insights to be gleaned from the text data.
<center> <a href="https://ibb.co/ZBKys0f"><img src="https://i.ibb.co/3pWtXnT/image.png" alt="image" border="0"></a> </center>
<center><p>Data Before Cleaning</p></center>

<center><p>Data After Cleaning</p></center>

<a href ="https://github.com/Taahaa-Dawe/OnlineDatingReview_TextMiningProject/blob/main/DataProceesed.csv"> Link to Cleaned Data</a>


	""",unsafe_allow_html=True)

		








	st.markdown(""" 
<div id="Code"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Code")
	st.write("""

	<ul>
	<li>Link to the Hierarchical clustering in R:  <a href = "https://github.com/Taahaa-Dawe/OnlineDatingReview_TextMiningProject/blob/main/Hclust.R">Hclust </a>
	<li>Link to the K-means clustering in python: <a href = "https://github.com/Taahaa-Dawe/OnlineDatingReview_TextMiningProject/blob/main/kmeansclustering.py"> K-means</a>

	""",unsafe_allow_html=True)

	st.markdown(""" 
<div id="Result"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Results")
	st.write("""
	<b>Hierarchical clustering:</b>
	<center> <a href="https://ibb.co/0MLySjM"><img src="https://i.ibb.co/fpWd5Yp/image.png" alt="image" border="0"></a> </center>
	<center>Dendrogram using Euclidean Distance</center>
	<br>
	<center> <a href="https://ibb.co/cY05Jph"><img src="https://i.ibb.co/NT5hxk2/image.png" alt="image" border="0"></a> </center>
	<center>Dendrogram using Cosine Similarity</center>
	The cosine similarity is not accurately reflecting the dendrogram, as it tends to cluster most of the data into one group, with only a single cluster forming in another part. This discrepancy likely arises from the nature of cosine similarity itself and its sensitivity to the vector angles between data points, rather than their magnitudes.

Cosine similarity measures the cosine of the angle between two vectors, representing the similarity in direction between them. However, it does not consider the magnitude of the vectors, meaning that vectors with different magnitudes can still have a high cosine similarity if they point in similar directions. In the context of clustering, this can lead to the grouping of data points that have similar directional trends but differ significantly in scale or magnitude.

For instance, if one cluster contains data points with relatively small magnitudes but similar directional trends, while another cluster consists of data points with larger magnitudes but different directional trends, cosine similarity may still perceive them as similar due to their directional alignment. This can result in the formation of larger, less meaningful clusters, rather than capturing the nuanced differences between distinct groups within the data.

In contrast, Euclidean distance, which considers both the direction and magnitude of vectors, often provides a more accurate representation of the data's clustering structure. It accounts for the overall distance between data points in multidimensional space, allowing for the identification of more distinct clusters based on both their spatial proximity and magnitude differences. As a result, Euclidean distance may better capture the underlying patterns and relationships within the data, leading to a more informative dendrogram visualization.
	<br><br>
	<b>K Means Clustering: </b>	
<center>
  <a href="https://imgbb.com/"><img src="https://i.ibb.co/GRw6szT/image.png" alt="image" border="0"></a>
</center>
<center><p>Distance Matrix for First Few Rows</p></center>
<br>
<center>
<a href="https://imgbb.com/"><img src="https://i.ibb.co/9nsvXtz/image.png" alt="image" border="0"></a>
</center>
<center>Word Cloud of Cluster One</center><br>
Cluster 1 portrays the landscape of modern dating and relationships, heavily influenced by digital platforms and apps. The inclusion of terms like "app," "Bumble," and "Hinge" reflects the prevalence of online dating in contemporary society. The words "meet," "date," and "profile" suggest the initial stages of forming connections, highlighting the importance of digital profiles and virtual interactions in sparking real-life encounters. Additionally, phrases such as "happy," "married," and "couple" signify the ultimate goal for many individuals navigating these platforms – finding lasting companionship or love. However, there's also a recognition of the challenges inherent in this process, as indicated by phrases like "do not," "doesn't," and "really." These words hint at the complexities, uncertainties, and occasional disappointments that can accompany the quest for meaningful connections in the digital age.
<br><br>
<center>
<a href="https://imgbb.com/"><img src="https://i.ibb.co/z84c7pH/image.png" alt="image" border="0"></a>
</center>
<center>Word Cloud of Cluster Two</center><br>
In contrast, Cluster 2 delves into the dynamics and evolution of relationships themselves. Words like "relationship," "talked," and "experience" emphasize the importance of communication and shared interactions in building and maintaining romantic connections. The presence of terms such as "recently," "started," and "moved" suggests a focus on the progression of relationships over time, highlighting key milestones and transitions. The inclusion of specific app names like "Tinder" and "Bumble" underscores the role of digital platforms in facilitating these connections, reflecting the increasingly intertwined nature of technology and romance. Overall, Cluster 2 paints a picture of individuals navigating the complexities of modern relationships, from the initial spark of attraction to the deepening of emotional bonds, all within the context of a digital dating landscape.
<br><br>
	

<center>
 <a href="https://imgbb.com/"><img src="https://i.ibb.co/HCLZbPp/image.png" alt="image" border="0"></a>
</center>
<center>Silhouette value for K=2</center> <br>
<b>Different Values of K and their Silhouette Score: </b> <br><br>
<center> <a href="https://ibb.co/c34f9jW"><img src="https://i.ibb.co/zxq9gz3/image.png" alt="image" border="0"></a> </center> 
<center>Silhouette value for K=3</center> <br>
<center> <a href="https://ibb.co/1ZF31c0"><img src="https://i.ibb.co/sj8hn7g/image.png" alt="image" border="0"></a> </center> 
<center>Silhouette value for K=4</center> <br>
<center> <a href="https://ibb.co/syjQznD"><img src="https://i.ibb.co/m6JtgkY/image.png" alt="image" border="0"></a> </center> 
<center>Silhouette value for K=5</center> <br>
<center> <a href="https://ibb.co/bFbt8Pb"><img src="https://i.ibb.co/m0NpMGN/image.png" alt="image" border="0"></a> </center> 
<center>Silhouette value for K=6</center> <br>

<center> <a href="https://imgbb.com/"><img src="https://i.ibb.co/SccLBcz/image.png" alt="image" border="0"></a> </center>
<center> Comparison of number of clusters and their Silhouette scores </center>

The silhouette analysis suggests that the maximum silhouette average value occurs at 2, indicating that this could be considered the optimal number of clusters.
<br>
<center>  <a href="https://imgbb.com/"><img src="https://i.ibb.co/TkPLtgp/image.png" alt="image" border="0"></a> </center>
<center> WCSS (Elbow Method) </center>

The elbow method also suggests that the optimal number of clusters is 2


Comparison of Hierarchical Clustering and K-means:

In the given scenario, both hierarchical clustering (hclust) with cosine similarity and K-means clustering suggested two clusters. However, when using cosine similarity with hierarchical clustering, it didn't effectively separate the data into distinct groups. On the other hand, using Euclidean distance with hierarchical clustering suggested two clusters, indicating a clearer separation of the data. This discrepancy suggests that the choice of distance metric significantly impacts the clustering results.

Overall, the fact that both hierarchical clustering with Euclidean distance and K-means clustering suggested two clusters implies that the underlying structure of the data likely consists of two distinct groups or classes. This demonstrates the importance of selecting appropriate distance metrics and clustering algorithms based on the characteristics of the data to achieve meaningful clustering results.
	""",unsafe_allow_html=True)
	st.markdown(""" 
<div id="Conclusion"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)

	st.subheader("Conclusion")
	st.write("""
The analysis dives into the complexities of modern dating and relationships, shining a light on how digital platforms and apps like "Bumble" and "Hinge" influence the way people connect. It emphasizes how online interactions through terms like "meet," "date," and "profile" play a big role in initiating real-life meetings.

Additionally, it explores the dynamics within relationships, highlighting the importance of communication and shared experiences, as seen in words like "relationship," "talked," and "experience." It also shows how relationships evolve over time, with phrases like "recently," "started," and "moved."

Overall, the analysis gives us a clear picture of how technology impacts romance, offering both opportunities and challenges. It helps us understand the modern dating scene, from initial digital sparks to deeper emotional connections, all within today's digitally connected world.

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

def LDA():
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
	elif option =="LDA":
		LDA()
	else:
		st.write("Work in Progress")


st.title("Exploring the Landscape of Modern Romance through Text Mining and Machine Learning Algorithms")


header = st.container()
with header:
	option = st.selectbox("", ("Introduction", "DataPrep/EDA", "Clustering", "ARM","LDA","DT","NB","SVM"))
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
