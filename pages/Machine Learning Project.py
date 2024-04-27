import streamlit as st
import pandas as pd
from PIL import Image
import streamlit.components.v1 as components
option = ""

def select(option):
	DisplayPart(option)

def SVM():
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
	st.header("Support Vector Machine", divider = "blue")

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
Support Vector Machines are a powerful and versatile supervised learning algorithm primarily used for classification tasks, but they can also be used for regression and outlier detection. 
<b>Here's a brief overview of how SVM works:</b>
<ul>
<li> <b> Margin: </b> In SVM, the goal is to find the hyperplane that maximizes the margin, which is the distance between the hyperplane and the nearest data points (support vectors) from each class. This margin maximization helps SVMs generalize well to unseen data.
<li> <b>  Hyperplane: </b> A hyperplane is a decision boundary that separates data points of different classes in a feature space. For a binary classification problem, the hyperplane is a (d-1)-dimensional subspace of the d-dimensional feature space.
<li> <b>  Support Vectors: </b> Support vectors are the data points closest to the hyperplane and have a non-zero weight in determining the position of the hyperplane. These are critical for defining the decision boundary and optimizing the margin.
<li> <b> Kernel Trick: </b> The kernel trick allows SVMs to implicitly map input data into a higher-dimensional space where a linear separation boundary can be applied. This transformation enables SVMs to handle non-linear decision boundaries effectively without explicitly calculating the coordinates of the data in the higher-dimensional space.
<li> <b> Optimization: </b> SVM formulates the problem of finding the optimal hyperplane as a convex optimization problem. The objective is to minimize a cost function, which includes a regularization term to control the trade-off between maximizing the margin and minimizing classification errors. </ul>
<a href="https://ibb.co/nbxbBtQ"><img src="https://i.ibb.co/2yGySHc/image.png" alt="image" border="0"></a>
<center> IMB </center> <br>

<h4>Types of SVM:</h4>
<ul>
<li> <b> Linear SVM: </b> This is the standard form of SVM where the decision boundary is a straight line (or hyperplane in higher dimensions) that separates classes. Linear SVM works well when the data is linearly separable.
<center> <a href="https://imgbb.com/"><img src="https://i.ibb.co/DbY8q6s/image.png" alt="image" border="0"></a></center>
<center> <a href =”https://www.geeksforgeeks.org/support-vector-machine-algorithm/”> geeksforgeeks</a> </center> <br>

<center> <a href="https://imgbb.com/"><img src="https://i.ibb.co/XL7ysNy/image.png" alt="image" border="0"></a></center>
<center> <a href =”https://www.geeksforgeeks.org/support-vector-machine-algorithm/”> geeksforgeeks</a> </center><br>
<li> <b> Non-linear SVM: </b>  SVM can handle non-linearly separable data by using kernel functions such as polynomial, radial basis function (RBF), or sigmoid kernels. These kernels implicitly map the input data into higher-dimensional spaces where linear separation is possible.
<center> <a href="https://ibb.co/vj35H9Z"><img src="https://i.ibb.co/4V8y7kP/image.png" alt="image" border="0"></a></center>

<center> <a href =”https://www.dataspoof.info/post/what-is-support-vector-machine-learn-to-implement-svm-in-python/”> DataSpoof </a></center><br>

<li> <b> Multi-class SVM: </b> SVM inherently supports binary classification. To handle multi-class classification tasks, techniques such as one-vs-one or one-vs-all can be used. In one-vs-one, a separate SVM is trained for each pair of classes, while in one-vs-all, one SVM is trained for each class against all other classes.
<li> <b> Probabilistic SVM: </b>  Traditional SVM provides a binary classification decision. Probabilistic SVM extends SVM to provide probability estimates for class membership, allowing users to gauge the confidence of the classification decision.
<li> <b> Sequential Minimal Optimization (SMO): </b> SMO is a popular algorithm for training SVMs. It breaks down the optimization problem into smaller subproblems, making it computationally efficient, particularly for large datasets. </ul>
Support Vector Machines (SVMs) can be used effectively to differentiate between three sentiments—positive, neutral, and negative—regarding online dating. Positive sentiments reflect satisfaction, excitement, and successful experiences. Neutral sentiments entail factual statements or observations without a strong emotional tone. Negative sentiments encompass dissatisfaction, disappointment, and concerns regarding online dating experiences. SVMs analyze data to categorize sentiments accurately, facilitating a nuanced understanding of public perception toward online dating platforms.
<center>
<a href="https://ibb.co/qNdXQdV"><img src="https://i.ibb.co/QPfzwft/image.png" alt="image" border="0"></a>
</center>
<center>
	Example of  polynomial Kernal 
</center><br>

	""", unsafe_allow_html=True)












	st.markdown(""" 
<div id="DataPrep"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Data Prep")
	st.write("""
		The data was already structured in a format suitable for SVM classification. Here's a glimpse of what the data looks like:
	<center>
<a href="https://ibb.co/XF73Jky"><img src="https://i.ibb.co/Dw98KgG/image.png" alt="image" border="0"></a>
</center>
<center> Data</center><br>

The dataset was divided into two subsets: a training dataset comprising 70% of the data and a testing dataset comprising the remaining 30%. This partitioning ensures that the model is trained on one set of data and tested on another, disjoint set, preventing any potential bias in the evaluation process. The model was trained using the training dataset, and its performance was assessed using the testing dataset. This approach helps to provide an unbiased evaluation of the model's performance on unseen data.
<center>
<a href="https://ibb.co/xq1D6Y5"><img src="https://i.ibb.co/B2ytBGs/image.png" alt="image" border="0"></a>
</center> 
<center> Train Data</center><br>

<center>
<a href="https://ibb.co/f06sjG3"><img src="https://i.ibb.co/h8QS31G/image.png" alt="image" border="0"></a>
</center>
<center> Test Data</center>
<ul>
<li> <a href ="https://github.com/Taahaa-Dawe/OnlineDatingReview_TextMiningProject/blob/main/Wholedata.csv"> Link to Data </a>

</ul>


	""",unsafe_allow_html=True)








	st.markdown(""" 
<div id="Code"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Code")
	st.write("""
	<li> <a href ="https://github.com/Taahaa-Dawe/Machine_Learning_Project_AB_Testing/blob/main/svm.py">Code</a>
	""",unsafe_allow_html=True)

	st.markdown(""" 
<div id="Result"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Results")
	st.write("""
<b>Linear SVM with cost 10 </b>
<center> <a href="https://imgbb.com/"><img src="https://i.ibb.co/LRqzdH8/image.png" alt="image" border="0"></a> </center>
<center>Classification Report</center><br>
<center> <a href="https://imgbb.com/"><img src="https://i.ibb.co/RCBCZDG/image.png" alt="image" border="0"></a> </center>
<center>Confusion Matrix</center><br>
<b>Linear SVM with cost 40 </b>
<center> <a href="https://imgbb.com/"><img src="https://i.ibb.co/5cYz7hT/image.png" alt="image" border="0"></a> </center>
<center>Classification Report</center><br>
<center> <a href="https://imgbb.com/"><img src="https://i.ibb.co/GPC2BzT/image.png" alt="image" border="0"></a> </center>
<center>Confusion Matrix</center><br>
<b>RBF SVM with cost 1 </b>
<center> <a href="https://imgbb.com/"><img src="https://i.ibb.co/hMRPSxV/image.png" alt="image" border="0"></a> </center>
<center>Classification Report</center><br>
<center> <a href="https://imgbb.com/"><img src="https://i.ibb.co/2SP4cw6/image.png" alt="image" border="0"></a> </center>
<center>Confusion Matrix</center><br>
<b>Poly SVM with cost 40 and degree 3 </b>
<center> <a href="https://imgbb.com/"><img src="https://i.ibb.co/TYdx9hq/image.png" alt="image" border="0"></a> </center>
<center>Classification Report</center><br>
<center> <a href="https://imgbb.com/"><img src="https://i.ibb.co/2SP4cw6/image.png" alt="image" border="0"></a> </center>
<center>Confusion Matrix</center><br>
<b>Poly SVM with cost 2 and degree 3 </b>
<center> <a href="https://imgbb.com/"><img src="https://i.ibb.co/BfqSj5X/image.png" alt="image" border="0"></a> </center>
<center>Classification Report</center><br>
<center> <a href="https://imgbb.com/"><img src="https://i.ibb.co/2SP4cw6/image.png" alt="image" border="0"></a> </center>
<center>Confusion Matrix</center><br>
<center> <a href="https://ibb.co/3M2rKCV"><img src="https://i.ibb.co/X312HC6/image.png" alt="image" border="0"></a> </center>
<center>Model Comparison</center><br>
The SVM (Support Vector Machine) model exhibits commendable performance in predicting campaign outcomes when utilizing a Linear kernel with a cost of 40, achieving the highest accuracy of 0.89. This indicates a strong fit for the dataset at hand. In contrast, the RBF and Polynomial kernels with varying cost parameters did not perform as effectively, consistently reaching an accuracy of 0.61. These results underscore the importance of kernel selection and hyperparameter tuning in SVM models to achieve optimal classification results.


	""",unsafe_allow_html=True)
	st.markdown(""" 
<div id="Conclusion"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)

	st.subheader("Conclusion")
	st.write("""
Throughout our analysis, we experimented with different settings on our predictive model to determine which would most accurately distinguish between two types of campaign strategies: Test and Control. We discovered that one particular setting stood out by providing the clearest and most accurate results, making it easier to understand which strategy might be more effective. While other settings were also explored, they did not perform as well, often mixing up the two types of campaigns or not identifying them as effectively. This exercise showed us that with the right adjustments, our model could accurately identify the best strategies, which could be incredibly useful for planning and decision-making in real-world scenarios.


""",unsafe_allow_html=True)

def DT():
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
	st.header("Decision Trees", divider = "blue")

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

Decision Trees (DTs) are versatile machine-learning models for classification and regression tasks. These hierarchical structures comprise nodes representing features, branches representing decision rules, and leaf nodes indicating class labels or numerical values. They are commonly employed in various applications such as spam detection, sentiment analysis, medical diagnosis, and predicting house prices. Decision trees operate by recursively partitioning the dataset based on feature values until certain stopping criteria are met, such as maximum depth or minimum samples per leaf. During inference, instances traverse the tree, with each node making decisions based on feature values until reaching a leaf node, where the final prediction is made. Supporting images, such as decision tree diagrams and decision boundaries, help illustrate their operation and effectiveness in partitioning the feature space for classification tasks.
<center><a href="https://imgbb.com/"><img src="https://i.ibb.co/fDzRydT/image.png" alt="image" border="0"></a> </center>
<center><a href = “https://medium.com/@nidhigh/decision-trees-a-powerful-tool-in-machine-learning-dd0724dad4b6”> Medium</a></center><br>
Key metrics used in decision tree algorithms include GINI impurity, entropy, and information gain. GINI impurity and entropy quantify the disorder or impurity of a dataset before and after a split, while information gain measures the reduction in impurity achieved by a particular split. For instance, in a binary classification scenario, the "goodness" of a split based on a feature like "Age" can be assessed by calculating the GINI impurity or entropy for the original dataset and each resulting subset, and then evaluating the information gain.
<br>
<center> <a href="https://ibb.co/zxQqgh4"><img src="https://i.ibb.co/3d4gbyC/image.png" alt="image" border="0"></a>
</center >
<center>Formulas </center><br>
One notable aspect of decision trees is their potential for infinite growth. Decision trees can theoretically split the feature space indefinitely, creating increasingly refined partitions. Moreover, there exist numerous parameters and hyperparameters that can be adjusted, such as maximum tree depth or minimum samples per leaf, offering a vast array of potential tree configurations. However, to mitigate the risk of overfitting and ensure manageable model complexity, various techniques such as pruning and regularization are commonly employed to control tree growth and generalize well to unseen data.

	""", unsafe_allow_html=True)












	st.markdown(""" 
<div id="DataPrep"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Data Prep")
	st.write("""
		The data was already structured in a format suitable for Decision Trees. Here's a glimpse of what the data looks like:

<br>
<center>
<a href="https://ibb.co/XF73Jky"><img src="https://i.ibb.co/Dw98KgG/image.png" alt="image" border="0"></a>
</center>
<center> Data</center><br><br>
The dataset was divided into two subsets: a training dataset comprising 70% of the data and a testing dataset comprising the remaining 30%. This partitioning ensures that the model is trained on one set of data and tested on another, disjoint set, preventing any potential bias in the evaluation process. The model was trained using the training dataset, and its performance was assessed using the testing dataset. This approach helps to provide an unbiased evaluation of the model's performance on unseen data. <br>
<br>
<center>
<a href="https://ibb.co/0VdvfvW"><img src="https://i.ibb.co/n85tDtV/image.png" alt="image" border="0"></a>
</center>
<center>Train Data</center><br>

<center>
<a href="https://ibb.co/bBhNKgY"><img src="https://i.ibb.co/KqZVrjB/image.png" alt="image" border="0"></a>
</center>
<center>Test Data</center>
<ul>
<li> <a href ="https://github.com/Taahaa-Dawe/OnlineDatingReview_TextMiningProject/blob/main/Wholedata.csv"> Link to Data </a>

</ul>
	""",unsafe_allow_html=True)








	st.markdown(""" 
<div id="Code"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Code")
	st.write("""
	<ul>
		<li> <a href = "https://github.com/Taahaa-Dawe/Machine_Learning_Project_AB_Testing/blob/main/Decision%20Trees.py"> Code</a>
	</ul>
	""",unsafe_allow_html=True)

	st.markdown(""" 
<div id="Result"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Results")
	st.write("""
<center>
<a href="https://ibb.co/PGNGr63"><img src="https://i.ibb.co/b5P56Kq/image.png" alt="image" border="0"></a>
</center>
<center> Descision Tree 1 </center> <br>
<center>
<a href="https://ibb.co/ZG07430"><img src="https://i.ibb.co/KbJS4YJ/image.png" alt="image" border="0"></a>
</center>
<center> Descision Tree 2 </center> <br>
<center>
<a href="https://ibb.co/mF1CnRQ"><img src="https://i.ibb.co/k6kgLGY/image.png" alt="image" border="0"></a> 
</center>
<center> Descision Tree 3 </center> <br>
<center>
<a href="https://imgbb.com/"><img src="https://i.ibb.co/VHmwXDn/image.png" alt="image" border="0"></a> 
</center>
<center> Confusion Matrix For 3rd tree</center><br>
<center>
<a href="https://imgbb.com/"><img src="https://i.ibb.co/FXJZLG7/image.png" alt="image" border="0"></a>
</center> 
<center> Classification  Report For 3rd tree</center><br>

With an accuracy of 83%, the model demonstrates proficiency in predicting the class labels, distinguishing between test campaign and control. This high accuracy suggests that the model performs well in correctly classifying instances into their respective categories based on the provided features.
	""",unsafe_allow_html=True)
	st.markdown(""" 
<div id="Conclusion"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)

	st.subheader("Conclusion")
	st.write("""
 Our decision tree model performed admirably in distinguishing between "Control Campaigns" and "Test Campaigns," achieving an accuracy of 83%. While it mostly got it right, there were a few instances where it confused the two types. Moving forward, we can fine-tune the model to better discern between these campaigns, improving its accuracy. Additionally, by analyzing users' spending habits, clicks, and other behaviors, we can gain insights into who they are and what they're interested in. This information helps us tailor our campaigns more effectively, ensuring that we're reaching the right audience with the right message.


""",unsafe_allow_html=True)

def NB():
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
	st.header("Naive Bayes", divider = "blue")

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

Naive Bayes is a simple yet powerful probabilistic classifier based on Bayes' theorem with an assumption of independence between features. It is commonly used in machine learning for various classification tasks due to its simplicity, efficiency, and effectiveness, especially in text classification and spam filtering. Here's an overview of Naive Bayes:
<ul>
<li> <b>Bayes' Theorem: </b> At the core of Naive Bayes is Bayes' theorem, which describes the probability of an event, based on prior knowledge of conditions that might be related to the event. Mathematically, it's represented as:
<center>
<a href="https://ibb.co/DtqkPk7"><img src="https://i.ibb.co/ZVZKRKh/image.png" alt="image" border="0"></a>
</center>
<li> <b> Naive Assumption: </b> Naive Bayes assumes that the presence of a particular feature in a class is independent of the presence of any other feature. This is a strong assumption and often unrealistic in real-world scenarios, but despite its simplification, Naive Bayes often performs well in practice, especially with text data.

<li> <b>Types of Naive Bayes: </b>
<ol>
   <li>  <b> Multinomial Naive Bayes: </b> 
		Multinomial Naive Bayes is a variant of the Naive Bayes algorithm specifically tailored for classification tasks where features are categorical and represent counts or frequencies, commonly seen in text classification scenarios. In this approach, each document is represented as a vector of term frequencies or counts, where each element corresponds to the occurrence of a specific term within the document. 

The algorithm computes the probability of observing each term given each class, typically denoted as P(t|c) , where (t) represents a term and (c) represents a class. These probabilities are estimated from the training data using techniques like Laplace smoothing to handle cases where certain terms may not occur in specific classes. 

During classification, Multinomial Naive Bayes employs Bayes' theorem to calculate the posterior probability of each class given the observed term frequencies. It selects the class with the highest posterior probability as the predicted class for the document. Despite its simplicity, Multinomial Naive Bayes is effective in text classification tasks due to its ability to handle large feature spaces efficiently and its robustness to irrelevant features. However, it's important to note that Multinomial Naive Bayes assumes independence between terms, which might not always be accurate in real-world scenarios. Additionally, it can be sensitive to zero-frequency issues, where a term doesn't occur in any document of a particular class. To address this, smoothing techniques are often employed, which involve adding a small value to the observed frequencies of each term to avoid zero probabilities and account for unseen data. These techniques help improve the robustness of the model and enhance its performance when dealing with sparse data or rare events.
<center>
<a href="https://ibb.co/QMpSJt2"><img src="https://i.ibb.co/2dWVFHD/image.png" alt="image" border="0"></a>
</center>

<center>
<a href =  “https://thatware.co/naive-bayes/”> thatware.co</a>
</center>

   <li> <b> Gaussian Naive Bayes: </b> 
Gaussian Naive Bayes is an extension of Naive Bayes for classification with continuous features following a Gaussian distribution assumption. During training, it estimates the mean and variance for each feature in each class. Then, it calculates the likelihood of observing feature values given each class using the Gaussian distribution formula. During classification, it computes posterior probabilities for each class based on observed feature values and selects the class with the highest probability as the prediction. While simple and efficient, it assumes feature normality, which may not always hold true, and may not perform well with non-Gaussian data.
<center>
<a href="https://ibb.co/NL3qnyz"><img src="https://i.ibb.co/4jfLgVx/image.png" alt="image" border="0"></a>
</center>
<center> <a href =” https://www. kdnuggets.com/2020/06/naive-bayes-algorithm-everything.html”> KDnuggets </a></center>

   <li> <b>Bernoulli Naive Bayes: </b> 
	Bernoulli Naive Bayes is a variant of the Naive Bayes algorithm designed for classification tasks where features are binary or Boolean. It is commonly applied in text classification scenarios where features represent the presence or absence of specific words or terms in documents. 

During training, Bernoulli Naive Bayes estimates the probability of observing each feature (term) given each class. It calculates the likelihood of a term occurring in documents belonging to each class based on the training data. Mathematically, this involves computing the probability of observing a term given a class using the training documents. 

When classifying a new document, Bernoulli Naive Bayes calculates the likelihood of observing the presence or absence of each term given each class. It combines these likelihoods with the prior probabilities of each class to compute the posterior probability of each class using Bayes' theorem. Specifically, it calculates the likelihood of the document's features given each class and multiplies them with the prior probability of each class. The class with the highest posterior probability is then assigned as the predicted class for the document. 

Overall, Bernoulli Naive Bayes offers a simple yet effective approach to text classification tasks with binary features. Despite its simplicity, it can perform well in practice, especially in scenarios where the features are binary-valued, such as spam filtering or sentiment analysis.
</ol>
<li> <b>Training Process: </b>
<ul>
   	<li>Given a dataset with labeled examples, Naive Bayes estimates the probabilities required by Bayes' theorem from the training data.
   	<li> For each class, it calculates the prior probability (the probability of each class).
   	<li> It then calculates the likelihood of each feature given each class.
   	<li>Finally, it uses these probabilities to make predictions on new data.
	</ul>
<li>  <b> Classification Process: </b>
<ul>
   	<li>When presented with a new example, Naive Bayes calculates the posterior probability of each class given the features using Bayes' theorem.
   	<li>It selects the class with the highest posterior probability as the predicted class for the example.
</ul>
<li> <b>  Advantages: </b>
	<ul>
   <li> Simple and easy to implement.
   <li> Works well with high-dimensional data.
   <li> Computationally efficient, particularly for large datasets.
   <li> Performs well even with the presence of irrelevant features.
</ul>
</ul>


	""", unsafe_allow_html=True)












	st.markdown(""" 
<div id="DataPrep"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Data Prep")
	st.write("""
	The data was already structured in a format suitable for Naive Bayes classification. Here's a glimpse of what the data looks like:
	<center>
<a href="https://ibb.co/XF73Jky"><img src="https://i.ibb.co/Dw98KgG/image.png" alt="image" border="0"></a>
</center>
<center> Data</center><br>

The dataset was divided into two subsets: a training dataset comprising 70% of the data and a testing dataset comprising the remaining 30%. This partitioning ensures that the model is trained on one set of data and tested on another, disjoint set, preventing any potential bias in the evaluation process. The model was trained using the training dataset, and its performance was assessed using the testing dataset. This approach helps to provide an unbiased evaluation of the model's performance on unseen data.
<center>
<a href="https://ibb.co/xq1D6Y5"><img src="https://i.ibb.co/B2ytBGs/image.png" alt="image" border="0"></a>
</center> 
<center> Train Data</center><br>

<center>
<a href="https://ibb.co/f06sjG3"><img src="https://i.ibb.co/h8QS31G/image.png" alt="image" border="0"></a>
</center>
<center> Test Data</center>
	
<ul>
<li> <a href ="https://github.com/Taahaa-Dawe/OnlineDatingReview_TextMiningProject/blob/main/Wholedata.csv"> Link to Data </a>

</ul>
	""",unsafe_allow_html=True)








	st.markdown(""" 
<div id="Code"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Code")
	st.write("""
	<ul>
		<li> <a href = "https://github.com/Taahaa-Dawe/Machine_Learning_Project_AB_Testing/blob/main/naive_bayes.py"> Code</a>
	</ul>
	""",unsafe_allow_html=True)

	st.markdown(""" 
<div id="Result"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Results")
	st.write("""
<center>
<a href="https://imgbb.com/"><img src="https://i.ibb.co/tH8dZdn/image.png" alt="image" border="0"></a>
</center>
<center> Confusion Matrix 
</center><br>

<center>
<a href="https://imgbb.com/"><img src="https://i.ibb.co/QNHRxMK/image.png" alt="image" border="0"></a>
</center>
<center> Classification Report
</center>
<br>


With an accuracy of 81%, the model demonstrates proficiency in predicting the class labels, distinguishing between test campaign and control. This high accuracy suggests that the model performs well in correctly classifying instances into their respective categories based on the provided features. 
	""",unsafe_allow_html=True)
	st.markdown(""" 
<div id="Conclusion"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)

	st.subheader("Conclusion")
	st.write("""

Multinomial Naive Bayes is a valuable tool for analyzing A/B testing data. It helps us uncover patterns in various metrics such as spending, impressions, reach, website clicks, searches, view content, add to cart, and purchases, depending on the group a customer belongs to. This means we can better understand how different groups behave and make predictions about which group a new data point might belong to based on its characteristics. In simpler terms, Multinomial Naive Bayes helps us make sense of the data from A/B tests and predict which group a customer is likely to fall into, based on how they interact with our offerings.
""",unsafe_allow_html=True)
	
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
The dataset was initially in a numerical format, requiring discretization or binning to convert it into categorical data suitable for association rule mining (ARM). In this process, the original data columns were discarded, retaining only the transactional records. Numeric values were transformed into discrete categories such as "low," "moderate," "high," or "very high," facilitating the extraction of meaningful associations between different items or categories within transactions.
<center> 
<a href="https://ibb.co/tMWqrN8"><img src="https://i.ibb.co/WWYkJdg/image.png" alt="image" border="0"></a>
</center> 
<center> 
	Data Before Cleaning
</center>
<br>
<center>
<a href="https://ibb.co/fp6H2gB"><img src="https://i.ibb.co/2tVdy9m/image.png" alt="image" border="0"></a>
</center>		
<center> 
	Final Transactional Format
</center>
	""",unsafe_allow_html=True)








	st.markdown(""" 
<div id="Code"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Code")
	st.write("""
	The dataset underwent preprocessing to convert it into a format suitable for association rule mining (ARM), where it was transformed into a basket format. This involved loading the data after discretization and removing unnecessary columns, focusing solely on transaction records. Following this preparation, association rule mining was performed on the dataset, along with visualizations to explore the discovered associations further. Below is the code showcasing the steps for data preparation, ARM, and visualization of the association rules.
	<br><br>
	Code: &nbsp;<a  href = "https://github.com/Taahaa-Dawe/Machine_Learning_Project_AB_Testing/blob/main/Association%20Rule%20Mining%20in%20R"> Association Rule Mining</a>

	""",unsafe_allow_html=True)
	st.markdown(""" 
<div id="Result"> 
<br><br><br><br><br>
</div>
""",  unsafe_allow_html=True)
	st.subheader("Results")
	st.write("""
	A threshold support of 0.1 and confidence of 0.9, a total of 30 rules were obtained.
	<br>
<center><a href="https://ibb.co/QcGQXgZ"><img src="https://i.ibb.co/rtJ0vjg/image.png" alt="image" border="0"></a></center>
	<center> Scatter Plot Of Confidence and Support <br/><br/></center>
	The support for most of the rules is relatively low but the confidence and lift are high for most of the rules.
	The Top Rules are as follow: <br>
<div style="text-align:center;">
    <a href="https://ibb.co/SvnBH7V"><img src="https://i.ibb.co/6sPFGJD/image.png" alt="image" style="border:0;"></a><br />
	By Support
</div>
    <br>
<div style="text-align:center;">
    <a href="https://ibb.co/yFck7FY"><img src="https://i.ibb.co/Jjwpgjy/image.png" alt="image" style="border:0;"></a><br />
    By Confidence
</div>
    <br>

<div style="text-align:center;">
    <a href="https://ibb.co/mBrnqrk"><img src="https://i.ibb.co/z5W3RWp/image.png" alt="image" style="border:0;"></a><br />
    By Lift
</div> <br>

Additional insights can be gleaned by examining the association rules, particularly those concerning the Control Campaign and Test Campaign, and their impact on various features such as spending and purchasing behavior. These rules provide valuable information about the relationships between different campaign types and customer behavior metrics, aiding in the optimization and refinement of marketing strategies.


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
	
The above graphs and rules gives the following conclusions about Customer Behavior in each Campaigns .
<li>Campaigns targeting low purchasing and moderate search behavior tend to result in low carting behavior.
<li>When low search behavior coincides with very high impressions, it often leads to very high reach.
<li>The presence of control campaigns is associated with low viewing and moderate reach.
<li>Certain combinations of campaign types and customer behavior metrics, such as low carting and moderate search, indicate a preference for test campaigns.
<li>Additionally, rules involving spending, impressions, reach, and website clicks shed light on the effectiveness of campaigns in influencing customer engagement and purchasing decisions.
<li>Overall, these insights can inform marketing strategies, helping businesses optimize their campaigns for better customer engagement and sales outcomes.


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
	elif option == "Clustering":
		Clustering()
	elif option == "ARM":
		ARM()
	elif option == "NB":
		NB()
	elif option == "DT":
		DT()
	elif option == "SVM":
		SVM()
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
