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
In contrast, Cluster 2 delves into the dynamics and evolution of relationships themselves. Words like "relationship," "talked," and "experience" emphasize the importance of communication and shared interactions in building and maintaining romantic connections. The presence of terms such as "recently," "started," and "moved" suggests a focus on the progression of relationships over time, highlighting key milestones and Transactionss. The inclusion of specific app names like "Tinder" and "Bumble" underscores the role of digital platforms in facilitating these connections, reflecting the increasingly intertwined nature of technology and romance. Overall, Cluster 2 paints a picture of individuals navigating the complexities of modern relationships, from the initial spark of attraction to the deepening of emotional bonds, all within the context of a digital dating landscape.
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

Association rule mining can be applied to text data containing opinions on online dating to uncover patterns and relationships between different aspects of the dating experience. By analyzing the opinions expressed in the text, association rule mining can identify frequent co-occurrences or associations between specific words or phrases. For example, it could reveal that positive sentiments towards a particular dating app are often associated with specific features mentioned in the reviews, such as ease of use or success in finding matches. Additionally, association rule mining could uncover interesting insights into user preferences and behaviors, such as the tendency for individuals who express dissatisfaction with one aspect of online dating to also mention specific challenges or frustrations they encounter. Overall, by applying association rule mining to text data on online dating opinions, valuable insights can be gained into the factors that influence users' perceptions and experiences in the digital dating landscape.
""", unsafe_allow_html=True)












	st.markdown(""" 
<div id="DataPrep"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Data Prep")
	st.write("""
	The data, initially in text format, underwent transformation using a TF-IDF vectorizer. This method converts text into numerical vectors, considering the importance of each word relative to its frequency across a set of documents.

	After this transformation, the data was converted into a Transactions format. Here, non-zero values were replaced by their corresponding column names or words. This made the data more suitable for Transaction analysis
	<center> <a href="https://ibb.co/Wt0dRQ9"><img src="https://i.ibb.co/4d2D0cy/image.png" alt="image" border="0"></a> </center>
	<center> Data In Vector Format </center> <br><br>
	<center> <a href="https://ibb.co/x7sH23J"><img src="https://i.ibb.co/2k6NF5g/image.png" alt="image" border="0"></a> </center>
	<center> Data In Transactionsal Format </center>
	""",unsafe_allow_html=True)








	st.markdown(""" 
<div id="Code"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Code")
	st.write("""


The dataset underwent preprocessing to convert it into a format suitable for association rule mining (ARM), where it was transformed into a basket format. This involved loading the data, vectorizing, and removing unnecessary columns, focusing solely on transaction records. Following this preparation, association rule mining was performed on the dataset, along with visualizations to explore the discovered associations further. Below is the code showcasing the steps for data preparation, ARM, and visualization of the association rules.

The data was converted into Transactions data using Python, and R was used to perform association rule mining.

<a href="https://github.com/Taahaa-Dawe/OnlineDatingReview_TextMiningProject/blob/main/DataCleaingForTransitionData.py">Python Code: Converting Data to Transactions Data</a>

<a href="https://github.com/Taahaa-Dawe/OnlineDatingReview_TextMiningProject/blob/main/ARMDating.R">R Code: Association Rule Mining</a>

	""",  unsafe_allow_html=True)
	st.markdown(""" 
<div id="Result"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Results")
	st.write("""
	A threshold support of 0.03 and confidence of 0.8, a total of 24 rules were obtained.<br><br>
	<a href="https://ibb.co/Kh3n6ny"><img src="https://i.ibb.co/r019k9y/image.png" alt="image" border="0"></a>
	<center>Scatter Plot Of Confidence and Support</center><br>

The support for most of the rules is relatively low but the confidence and lift are high for most of the rules. <br> 
The Top Rules are as follow:<br>

<center><a href="https://ibb.co/BKf2b9X"><img src="https://i.ibb.co/TqwM6sV/image.png" alt="image" border="0"></a></center>
<center>Rules Sorted By Support</center><br>
<center><a href="https://ibb.co/4tgTzNj"><img src="https://i.ibb.co/Rb40fHC/image.png" alt="image" border="0"></a></center>
<center>Rules Sorted By Confidence</center><br>
<center><a href="https://ibb.co/dLnBVfs"><img src="https://i.ibb.co/Yyv21N4/image.png" alt="image" border="0"></a></center>
<center>Rules Sorted By Lift</center> <br>


Additional insights can be gleaned by examining the association rules through network diagrams. These diagrams visually represent the relationships between items or variables in the dataset, allowing for a more intuitive understanding of the associations discovered through association rule mining.
	""",unsafe_allow_html=True)
	html_file = open("pages/ARMTextMiningRules.html", 'r', encoding='UTF-8')
	source_code = html_file.read()
	html_file.close()
	components.html(source_code, height=750)
	
	
	st.markdown(""" 
<div id="Conclusion"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)

	st.subheader("Conclusion")
	st.write("""
	The analysis of the data uncovers intriguing connections between online activity and dating interests. It's clear that individuals who use dating apps and engage in online interactions are more likely to show an interest in dating. Topics commonly discussed online, such as relationships, apps, and social media, are strongly linked with this interest. Furthermore, positive expressions and curiosity, as indicated by words like "like" and "know," often accompany a heightened interest in dating.

Moreover, the analysis highlights an intriguing relationship between Facebook and dating interests. It appears that individuals who mention Facebook in their online discussions are more inclined towards dating. This suggests that Facebook, being a widely used social platform, serves as a significant avenue for individuals to explore and express their romantic interests. Whether it's through connecting with potential partners, sharing relationship status updates, or discussing dating-related topics, Facebook seems to play a pivotal role in facilitating and reflecting dating behaviors in the digital age.


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
	st.header("Latent Dirichlet Allocation (LDA)", divider = "blue")

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
Latent Dirichlet Allocation (LDA) is a powerful statistical model for uncovering underlying themes or topics within a collection of documents. It operates on the assumption that each document is a mixture of topics, and each topic is characterized by a distribution of words. By iteratively assigning words to topics and updating topic distributions based on these assignments, LDA uncovers the latent structure of the documents. Through this process, LDA represents each document as a distribution over topics, and each topic as a distribution over words. This allows for a deeper understanding of the content and themes present in the corpus. LDA finds applications in various fields such as natural language processing, information retrieval, and document classification, aiding in tasks like summarization, categorization, and recommendation systems. Its ability to automatically identify topics without the need for labeled data makes it a valuable tool for exploring and analyzing large text datasets.
<center> <a href="https://ibb.co/D8q3Vhm"><img src="https://i.ibb.co/TbCzv6J/image.png" alt="image" border="0"></a> </center>

<center> <a href = “https://www.analyticsvidhya.com/blog/2021/06/part-2-topic-modeling-and-latent-dirichlet-allocation-lda-using-gensim-and-sklearn/”> Analytics Vidhya</a> </center>
<br>
Latent Dirichlet Allocation (LDA) is needed for several reasons:
<ul>
<li> <b>Topic Discovery:</b> LDA enables the automatic discovery of underlying topics or themes within a collection of documents. This is particularly useful when dealing with large corpora where manual inspection of each document is impractical.

<li> <b>Document Summarization:</b>  By identifying the topics present in documents, LDA facilitates document summarization. It allows for the creation of concise representations of documents based on their dominant themes, making it easier to understand and analyze large volumes of text.

<li> <b>Information Retrieval: </b> LDA assists in information retrieval tasks by providing a way to index and search documents based on their topics. This helps users find relevant documents more efficiently, as documents can be ranked or filtered based on their topical relevance.

<li> <b>Content Recommendation:</b>  LDA can be used to build recommendation systems by identifying topics of interest for users. By analyzing the topics present in documents that a user has interacted with, LDA can suggest other documents or resources on similar topics.

<li> <b>Content Analysis:</b>  LDA aids in content analysis by revealing patterns and trends within a corpus. It allows researchers and analysts to gain insights into the distribution of topics over time, across different sources, or within specific domains.

<li> <b>Feature Selection:</b>  In machine learning tasks, LDA can be used for feature selection by identifying the most relevant words or terms associated with each topic. This can help improve the performance of models by focusing on the most discriminative features.

</ul>
<center> <a href="https://ibb.co/QHH5nHp"><img src="https://i.ibb.co/R665P6Y/image.png" alt="image" border="0"></a> </center>
<center> <a href = “https://www.kdnuggets.com/2019/09/overview-topics-extraction-python-latent-dirichlet-allocation.html”> Kdnuggets </a> </center>
<br>
<b>LDA assumes a generative process for document creation:</b>
<li>	For each document, it first decides on the distribution of topics that the document will contain. This distribution is drawn from a Dirichlet distribution over topics.
<li>	For each word in the document, it then chooses a topic from the document's distribution of topics.
<li>	Finally, it selects a word from the chosen topic's distribution over words. <br><br>
The goal of LDA is to infer the underlying topic structure from the observed words in the documents. It does this by iteratively updating its parameters (i.e., the topic distributions for documents and the word distributions for topics) until it converges to a stable solution.
<br><br>
<center> <a href="https://ibb.co/bWYg0by"><img src="https://i.ibb.co/FxcbrHf/image.png" alt="image" border="0"></a> </center>
<center> <a href= "https://towardsdatascience.com/topic-modeling-with-latent-dirichlet-allocation-e7ff75290f8"> Towards Data Science</a></center>
<br><br>
Latent Dirichlet Allocation (LDA) can uncover key topics within the dating-related text data. By identifying prevalent themes like online dating experiences and preferences, LDA can provide structured insights. It can also reveal trends over time and enables segmentation for targeted strategies. LDA can also enhance our understanding of the dataset, guiding decision-making effectively.

""", unsafe_allow_html=True)












	st.markdown(""" 
<div id="DataPrep"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Data Prep")
	st.write("""
The text data was first converted using CountVectorizer, limiting features to a maximum of 200. Then, Latent Dirichlet Allocation (LDA) was applied to uncover key topics within the dataset. LDA analyzes the frequency of words to identify prevalent themes like online dating experiences and preferences. By using a limited set of features, we focus on the most relevant aspects of the data, enhancing the effectiveness of the LDA analysis.
<center> <a href="https://ibb.co/7yzS7yv"><img src="https://i.ibb.co/DKb13K4/image.png" alt="image" border="0"></a> </center>

<center> Data Before Cleaning </center>
<br>
<center> <a href="https://ibb.co/6Ywc0Bs"><img src="https://i.ibb.co/0q2pytJ/image.png" alt="image" border="0"></a> </center>
<center> Data After Cleaning </center>
	""",unsafe_allow_html=True)








	st.markdown(""" 
<div id="Code"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Code")
	st.write("""
	<li>  
	<a href = "https://github.com/Taahaa-Dawe/OnlineDatingReview_TextMiningProject/blob/main/LDA.py"> Latent Dirichlet Allocation (LDA) </a>
""",  unsafe_allow_html=True)
	st.subheader("Results")
	st.write("""
	
<center> <a href="https://ibb.co/hgKVvyM"><img src="https://i.ibb.co/kGhXFxy/image.png" alt="image" border="0"></a>  </center>
<center> Topic Viz </center>


Topic #0 seems to capture discussions around personal experiences and feelings related to dating. Words like "just," "like," "don't," and "want" suggest a more subjective or emotional aspect of dating, possibly reflecting individual sentiments and perspectives.

Topic #1, on the other hand, appears to be more about practical aspects of dating, such as the use of dating apps, interactions with potential partners, and considerations about profiles and matches.

In essence, both topics revolve around dating, but Topic #0 may lean towards subjective experiences and emotions, while Topic #1 focuses more on practical aspects and actions related to dating.

	""",unsafe_allow_html=True)
	html_file = open("pages/ViewsonDating.html", 'r', encoding='UTF-8')
	source_code = html_file.read()
	html_file.close()
	components.html(source_code, height=800, width = 600)
	
	st.markdown(""" 
<div id="Conclusion"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)

	st.subheader("Conclusion")
	st.write("""
	
The analysis uncovered two main themes within dating-related conversations. One theme revolves around personal experiences and emotions tied to dating, while the other focuses on practical aspects such as using dating apps and interacting with potential partners. These findings offer valuable insights into the various dimensions of discussions about dating, shedding light on both the emotional and practical considerations involved. Such understanding is crucial for navigating the complexities of modern relationships and enhancing communication in the digital dating landscape.

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
