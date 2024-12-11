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
	st.header("Data Preparation", divider='blue')
	st.write("""The data consists of two files. File 1, sourced from <a href ="https://fr.wikipedia.org/wiki/Tifinagh">Wikipedia</a>, contains 31 letters of the Tifinagh alphabet. File 2, sourced from <a href ="https://tatoeba.org/en/downloads">Tatoeba</a>, comprises Amazigh-English sentence pairs, originally totaling 15,371 rows before cleaning and the same number after cleaning. """,unsafe_allow_html=True)
	st.write("")
	st.write("""<center> 
 <a href="https://ibb.co/StTZDw9"><img src="https://i.ibb.co/fkRTVdP/image.png" alt="image" border="0"></a><br /><a target='_blank' href='https://imgbb.com/'>Figure 6: Tifinagh from Wikipedia with 31 letters in total.</a><br />
 	</center>""",unsafe_allow_html=True)
	st.write("")
	st.write("""<center> 
 	<a href="https://ibb.co/10tVG6D"><img src="https://i.ibb.co/TwxXMLV/image.png" alt="image" border="0"></a><br /><a target='_blank' href='https://imgbb.com/'>Figure 7: Sentence Pairs (Amazigh – English) from Tatoeba with 15,371 rows (before cleaning).</a><br />
  	</center>""",unsafe_allow_html=True)
	st.write("")
	st.write(""" No exact duplicates were found in File 2. However, duplicates in English translations (due to dialectal variations) and Amazigh translations were found and removed, resulting in the deletion of 4,701 rows, accounting for 30.58% of the data. The remaining sentences were standardized to include both romanized and Tifinagh scripts, with the romanization process performed using File 1. The Tamazight vocabulary in the cleaned dataset contains 7,161 unique words, compared to 3,929 unique English words.""",unsafe_allow_html=True)
	st.write("")
	st.write("""<center> <a href="https://ibb.co/Wcjc6Dy"><img src="https://i.ibb.co/wKjKRJW/image.png" alt="image" border="0"></a><br /><a target='_blank' href='https://imgbb.com/'>Figure 8: Sentence Pairs (Amazigh – English) from Tatoeba with 10,670 rows (after cleaning).</a><br /> </center>""",unsafe_allow_html=True)
	st.write("")
	st.subheader("Data Exploration")
	st.write("""<center><a href="https://ibb.co/YdY45JM"><img src="https://i.ibb.co/2Mm0zRQ/image.png" alt="image" border="0"></a><br /><a target='_blank' href='https://imgbb.com/'>Figure 9: English (Left) and Tamazight’s (Right) count of words in descending order.</a><br /></center>""",unsafe_allow_html=True)
	st.write("")
	st.subheader("English Sentences in the Dataset")
	st.write("""An analysis of the English sentences in the dataset reveals potential challenges for training. A high frequency of stopwords, pronouns, and people’s names such as “Tom” dominates the text. This imbalance might hinder the model’s ability to learn meaningful patterns, as these elements contribute little to understanding the structural and semantic nuances of the language. """,unsafe_allow_html=True)
	st.write("""<center><a href="https://imgbb.com/"><img src="https://i.ibb.co/Jd3GKKG/image.png" alt="image" border="0"></a><br /><a target='_blank' href='https://imgbb.com/'>Figure 10: WordCloud of English Sentences in the Dataset.</a><br /></center>""",unsafe_allow_html=True)
	st.write("")
	st.subheader("Tamazight Sentences in the Dataset")
	st.write("""A similar trend is observed in the Tamazight sentences, as demonstrated by a word cloud of the romanized Amazigh text. Stopwords and pronouns are prevalent, with “Tom” (romanized as “tum”) appearing frequently, although less so than in the English dataset. This overrepresentation poses a challenge for training, as it may skew the model’s focus away from more informative linguistic features.""",unsafe_allow_html=True)
	st.write("")
	st.write("""<center><a href="https://imgbb.com/"><img src="https://i.ibb.co/8D0mfkW/image.png" alt="image" border="0"></a><br /><a target='_blank' href='https://imgbb.com/'>Figure 11: WordCloud of English Sentences in the Dataset</a><br /></center>""",unsafe_allow_html=True)
	st.write("")
	st.subheader("Data Prep and EDA:")
	st.write("""https://drive.google.com/drive/folders/1QvSNVwY0pc176zVzGE9nJ_hYTw3n23py?usp=share_link """,unsafe_allow_html=True)
	
def Methodology():
	st.header("Methodology", divider='blue')
	st.subheader("Data Preparation")
	st.write("""The first step in the process was to collect and prepare the data for training the models. A parallel dataset containing sentence pairs in Amazigh (as input) and English (as output) was sourced to provide the foundation for the translation task. This dataset served as the basis for both training and evaluating the models’ performance.""")
	st.write("""<center>
 <a href="https://ibb.co/VBVzxZd"><img src="https://i.ibb.co/RH2sNm1/image.png" alt="image" border="0"></a><br /><a target='_blank' href='https://imgbb.com/'>Figure 12: Sample from cleaned dataset with Tamazight in Tifinagh (left) and English (right).</a><br />
 	</center>""",unsafe_allow_html=True)
	st.write("")
	st.write("""Data cleaning was a critical part of the preparation process. Duplicate and irrelevant entries were removed to improve the quality and relevance of the dataset. The Amazigh text was standardized by replacing Amazigh characters with their Romanized equivalents, as the models lacked support for the Tifinagh script. Additionally, punctuation and formatting inconsistencies were cleaned to ensure uniformity across the dataset, reducing potential noise during training. """,unsafe_allow_html=True)
	st.write("""<center><a href="https://imgbb.com/"><img src="https://i.ibb.co/ys4wXHJ/image.png" alt="image" border="0"></a><br /><a target='_blank' href='https://imgbb.com/'>Figure 13: Sample from cleaned dataset with Romanized Tamazight.</a><br /></center>""",unsafe_allow_html=True)
	st.write("")
	st.write("""Tokenization was then applied to break the sentences into smaller units, or tokens, which enabled the models to process them effectively. Each token was converted into a numerical representation suitable for input to the machine learning models. To maintain uniform input lengths, sentences were padded, meeting the structural requirements for training. This data preparation process ensured the dataset was optimized for use in building accurate and reliable translation models.""",unsafe_allow_html=True)
	st.write("")
	st.write("""<center><a href="https://imgbb.com/"><img src="https://i.ibb.co/Bzx9DNy/image.png" alt="image" border="0"></a><br /><a target='_blank' href='https://imgbb.com/'>Figure 14: Sample of tokenized and padded data.</a><br /></center>""",unsafe_allow_html=True)
	st.write("")
	st.subheader("Models and Techniques")
	st.write("""To translate Amazigh to English, several models and techniques were employed. The Sequence-to-Sequence (Seq2Seq) model served as the baseline, leveraging an encoder-decoder architecture. The encoder processed Amazigh input sentences and converted them into a “context vector,” which the decoder then used to predict English translations one word at a time. While this model was effective for simpler sentences, it struggled with longer and more complex structures due to its limited ability to capture long-range dependencies.""",unsafe_allow_html=True)
	st.write("")
	st.write("""<center> 
 <a href="https://ibb.co/bPWc0wv"><img src="https://i.ibb.co/nbgxGd1/image.png" alt="image" border="0"></a><br /><a target='_blank' href='https://imgbb.com/'>Figure 15: Architecture of a Sequence-to-Sequence Model (Github).</a><br />
 </center>""",unsafe_allow_html=True)
	st.write("")
	st.write(""" Transformer-based models were also implemented to handle the translation task more effectively. A custom Transformer model was built from scratch, utilizing the attention mechanism to focus on relevant parts of input sentences during translation. The model incorporated key components such as multi-head attention, which helped identify relationships between words, positional encoding to ensure the model recognized the order of words, and feed-forward neural networks to enhance encoded representations for accurate translations. This custom model improved translation quality but required extensive training on the dataset""",unsafe_allow_html=True)
	st.write(""" <center><a href="https://ibb.co/2NRHRKm"><img src="https://i.ibb.co/k1zvzck/image.png" alt="image" border="0"></a><br /><a target='_blank' href='https://imgbb.com/'>Figure 16: Architecture of a Transformer Model (Attention is All you Need).</a><br /></center>""",unsafe_allow_html=True)
	st.write("")
	st.write("""Additionally, a pre-trained Transformer model, Helsinki-NLP/opus-mt-en-ro from Hugging Face, was fine-tuned for the Amazigh-English translation task. Pre-trained on multilingual datasets, this model was faster and more efficient to adapt, as it already had a robust understanding of English. Fine-tuning allowed it to specialize in translating Amazigh sentences while maintaining high translation accuracy and fluency, outperforming the custom Transformer in both efficiency and output quality.""",unsafe_allow_html=True)
	st.write("""The Google Translate API was used as a benchmark for comparison with the custom and pre-trained models. Translation requests were sent via API calls to the service, which performed well for general translations. However, the API struggled with Amazigh-specific linguistic and cultural nuances, highlighting the limitations of generalized translation tools for low-resource languages.""",unsafe_allow_html=True)
	st.write("")
	st.subheader("Evaluation")
	st.write("""The models were evaluated based on the quality and accuracy of their translations, with BLEU scores and accuracy metrics used as key measures. The BLEU score assessed translation quality by comparing the models’ outputs with reference translations. The custom Transformer model faced prediction issues, making it challenging to calculate a BLEU score. However, the fine-tuned pre-trained Transformer from Hugging Face achieved a BLEU score of 49.5, reflecting high-quality translations. In comparison, the Google Translate API yielded a BLEU score of 14.9, highlighting its struggles with Amazigh-specific linguistic nuances.""",unsafe_allow_html=True)
	st.write("""Accuracy was also evaluated to determine the percentage of correct predictions. Among the models, the pre-trained Transformer achieved the highest accuracy at approximately 96%, demonstrating its robustness and ability to produce reliable translations. This performance significantly outpaced both the custom Transformer and the Google Translate API, emphasizing the effectiveness of fine-tuning pre-trained models for low-resource language tasks.""",unsafe_allow_html=True)

	
def AnalysisResults():
	st.header("Analysis and Results")
	st.subheader("Model 1: Sequence to sequence")
	st.write("""The Seq2Seq model’s performance varied significantly across different configurations of batch size and training epochs, influencing its ability to learn and generalize effectively. With a batch size of 64 and 150 epochs (Figure 17), the model achieved a training accuracy of 97% and a validation accuracy of 96%. This configuration allowed the model to handle simpler sentences reasonably well but struggled with more complex structures, such as those involving negations or rare linguistic patterns. The test loss evened out at higher values, indicating overfitting and limited generalization despite extended training. The smaller batch size enabled the model to capture finer details but caused noisier gradients, leading to occasional fluctuations in training accuracy. For longer or more nuanced sentences, the model often produced grammatically incomplete or redundant translations.""",unsafe_allow_html=True)
	st.write(""" <center> <a href="https://ibb.co/FB3Mk8H"><img src="https://i.ibb.co/XzLgfFY/image.png" alt="image" border="0"></a><br />Figure 17: Accuracy (left) and loss (right) plots for Seq2Seq with 64 Batches and 150 Epochs.<br /> </center>""",unsafe_allow_html=True)
	st.write("")
	st.write("""When trained with a batch size of 100 and 150 epochs (Figure 18), the Seq2Seq model showed similar accuracy trends but with slightly slower convergence due to the larger batch size. While the larger batch size smoothed the learning process, it failed to capture nuanced linguistic patterns effectively, resulting in less fluent predictions for complex sentences. Test loss remained high, reflecting overfitting and reduced sensitivity to finer linguistic details.""",unsafe_allow_html=True)
	st.write(""" <center><a href="https://ibb.co/PDX3sKM"><img src="https://i.ibb.co/Z24tR5S/image.png" alt="image" border="0"></a><br />Figure 18: Accuracy (left) and loss (right) plots for Seq2Seq with 100 Batches and 150 Epochs.<br /> </center>""",unsafe_allow_html=True)
	st.write("")
	st.write("""With a shorter training duration of 50 epochs and a batch size of 64 (Figure 19), both training and validation accuracy evened out early, resulting in suboptimal learning. The test loss was higher compared to the 150-epoch runs, and the model struggled to generalize effectively. This configuration limited the Seq2Seq model’s ability to learn complex sentence structures, leading to inconsistent predictions, particularly for inputs requiring a deeper understanding of grammar and syntax.""",unsafe_allow_html=True)
	st.write("""<center><a href="https://ibb.co/pdYc5WT"><img src="https://i.ibb.co/gJkNhv8/image.png" alt="image" border="0"></a><br />Figure 19: Accuracy (left) and loss (right) plots for Seq2Seq with 64 Batches and 50 Epochs<br /></center>""",unsafe_allow_html=True)
	st.write("")
	st.write("""In summary, extended training with smaller batch sizes (64 and 150 epochs) provided better generalization for simpler sentences, while larger batch sizes (100) offered smoother training but reduced performance on complex linguistic patterns. Shorter training durations (50 epochs) significantly hindered the model’s ability to handle intricate translations.""",unsafe_allow_html=True)
	st.subheader("Model 2: Transformer")
	st.write("""The Transformer model shown here demonstrates strong training and validation metrics but fails to generate meaningful translations for the Amazigh-English task. The accuracy and loss graphs (Figure 20) indicate that the model achieved <b>training accuracy near 99.9%</b> and <b>validation accuracy close to 98%</b> after 18 epochs, with the loss reducing steadily during training. The <b>training loss dropped to nearly 0.017</b>, and the <b>validation loss plateaued at approximately 0.07</b>, suggesting that the model learned patterns from the training data effectively.""",unsafe_allow_html=True)
	st.write("""<center><a href="https://ibb.co/SvVLsTP"><img src="https://i.ibb.co/kcqdHnx/image.png" alt="image" border="0"></a><br />Figure 20: Accuracy (left) and loss (right) plots for the Transformer Model.<br /></center>""",unsafe_allow_html=True)
	st.write("")
	st.write("""However, despite these promising metrics, the model struggled to predict any coherent translations. When tested on new Amazigh sentences (in Tifinagh script) (Figure 21), the model failed to produce any meaningful English outputs. This indicates that while the model optimized its parameters during training, it likely overfitted to the training data and did not generalize well to unseen inputs. The lack of fine-tuning and limited training data likely contributed to its inability to capture the linguistic and syntactic nuances required for accurate translation.""",unsafe_allow_html=True)
	st.write("""<center><a href="https://ibb.co/WzLqY3Q"><img src="https://i.ibb.co/3FnGLsw/image.png" alt="image" border="0"></a><br />Figure 21: Transformer failing to make any translation.<br /></center>""",unsafe_allow_html=True)
	st.write("")
	st.write("""This highlights the importance of fine-tuning pre-trained models on domain-specific data and the challenges faced when training a model from scratch with limited resources.""",unsafe_allow_html=True)
	st.subheader("Model 3: Helsinki-NLP/opus-mt-en-ro")
	st.write("""The fine-tuned <b>Helsinki-NLP/opus-mt-en-ro model</b> demonstrated exceptional performance in the Amazigh-to-English translation task. The model was trained on Romanized inputs due to the lack of support for the Tifinagh script, yet it effectively processed the dataset and delivered high-quality translations. Over 10 epochs (Figure 22), the training loss consistently decreased from 0.1271 to 0.0143, while the validation loss steadily dropped from 0.1181 to 0.0706. This indicates that the model not only fits the training data well but also generalizes effectively to unseen data. The average BLEU score of 49.27 further highlights the model’s capability to produce grammatically correct and semantically accurate translations.""",unsafe_allow_html=True)
	st.write("""<center><a href="https://ibb.co/s1rb74b"><img src="https://i.ibb.co/dpyjZvj/image.png" alt="image" border="0"></a><br />Figure 22: Training and testing loss of the Helsinki-NLP model.<br /></center>""",unsafe_allow_html=True)
	st.write("")
	st.write("""The loss trends show that the model converged smoothly, with validation loss plateauing around epoch 6 (~0.07). This suggests that additional improvements might require a larger dataset or longer fine-tuning. Despite being trained on Romanized inputs, the model preserved much of the linguistic integrity of Amazigh, although the absence of Tifinagh script support may have limited its ability to capture cultural and idiomatic nuances. The use of pre-trained MarianMT weights enabled efficient learning, significantly outperforming from-scratch training methods like Seq2Seq and basic Transformers.""",unsafe_allow_html=True)
	st.write("""While the model excelled in both accuracy and generalization, there is room for improvement. The lack of Tifinagh script support remains a notable limitation, potentially impacting the model’s ability to handle idiomatic expressions and rare terms. Nevertheless, the Helsinki-NLP model sets a strong benchmark for low-resource language translation, and its performance reflects the effectiveness of fine-tuning pre-trained models for specific tasks. Future work could focus on integrating Tifinagh script support to further enhance linguistic fidelity and translation quality.""",unsafe_allow_html=True)
	st.subheader("Model 4: Google Translate API")
	st.write("""The Google Translate API served as a baseline for Amazigh-English translation, achieving an <b>average BLEU score of 14.55</b>. While it performed adequately for simple sentences, it struggled with complex structures, idiomatic expressions, and cultural nuances. The BLEU score distribution (Figure 23) showed significant variability, with many translations failing to align meaningfully with the reference sentences, as indicated by the 25th percentile BLEU score of 0.""",unsafe_allow_html=True)
	st.write("""<center><a href="https://imgbb.com/"><img src="https://i.ibb.co/0fx9G9m/image.png" alt="image" border="0"></a><br />Figure 23: BLEU score distribution of Google Translated Tamazight.<br /></center>""",unsafe_allow_html=True)
	st.write("")
	st.write("""The API’s general-purpose design and reliance on Romanized inputs limited its ability to handle the intricacies of Amazigh. Although it provided a fast solution for basic translations, its lack of specialization highlights the need for fine-tuned models for low-resource languages.""",unsafe_allow_html=True)
	st.subheader("Summary of all Models")
	st.write("""The table summary highlights the comparative performance of four models for translating Amazigh. The Sequence-to-Sequence and Transformer models showed limited success, with low prediction accuracy and some confusion in outputs. Google Translate API struggled, failing to generate predictions due to a lack of training for rare languages like Amazigh. """,unsafe_allow_html=True)
	st.write("""The Helsinki-NLP/opus-mt-en-ro model, hosted on Hugging Face, outperformed others, achieving a BLEU score of 49.27 and delivering more accurate results, albeit with minor confusion. In terms of Tifinagh script support, the Sequence-to-Sequence and Helsinki-NLP models accommodated it effectively, while the Transformer model provided partial support, and Google Translate failed to adapt to it. These findings emphasize the need for fine-tuned solutions for low-resource languages.""",unsafe_allow_html=True)
	st.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison Table</title>
    <style>
        body {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            font-family: Arial, sans-serif;
            background-color: #121212;
            color: #f5f5f5;
        }
        table {
            width: 80%;
            border-collapse: collapse;
            margin: 20px auto;
            background-color: #1e1e1e;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        th, td {
            border: 1px solid #444;
            text-align: left;
            padding: 12px;
            color: #f5f5f5;
        }
        th {
            background-color: #2c2c2c;
        }
        tr:nth-child(even) {
            background-color: #252525;
        }
        tr:hover {
            background-color: #333;
        }
        caption {
            margin-top: 10px;
            font-size: 14px;
            color: #bbb;
        }
    </style>
</head>
<body>
    <table>
        <caption><center>Table 1: Summary Table of results of the four models.</center></caption>
        <thead>
            <tr>
                <th>Model / KeyPoints</th>
                <th>Sequence-to-sequence</th>
                <th>Transformer model</th>
                <th>Google Translate API</th>
                <th>Helsinki-NLP</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Predictions</td>
                <td>Few correct words predicted.</td>
                <td>No predictions.</td>
                <td>Good predictions, some confusion.</td>
                <td>Good predictions, some confusion.</td>
            </tr>
            <tr>
                <td>Blue Score (%)</td>
                <td>0</td>
                <td>–</td>
                <td>14.55</td>
                <td>49.27</td>
            </tr>
            <tr>
                <td>Accuracy (%)</td>
                <td>98</td>
                <td>96</td>
                <td>–</td>
                <td>96</td>
            </tr>
            <tr>
                <td>Tifinagh Support</td>
                <td>Yes</td>
                <td>Unknown (Did not learn)</td>
                <td>Yes</td>
                <td>No</td>
            </tr>
        </tbody>
    </table>
</body>
</html>

""",unsafe_allow_html=True)
	#st.write("<center>Table 1: Summary Table of results of the four models.</center>",unsafe_allow_html=True)
	st.write("")
	st.write("""To sum up, the Helsinki-NLP/opus-mt-en-ro Transformer model, hosted on Hugging Face, demonstrated superior performance compared to traditional approaches in translating Amazigh to English. While traditional Seq2Seq models struggled with the complexity of linguistic nuances, and Google Translate failed to provide adequate specificity for rare languages, the fine-tuned Transformer model (Helsinki-NLP) effectively captured linguistic patterns. It delivered high-quality translations, making it a robust solution for addressing Amazigh-English translation challenges. """,unsafe_allow_html=True)
	st.write("""Recent efforts indicate ongoing initiatives to enhance Tamazight language support in tools like Google Translate, further underscoring the importance of advancing translation technologies for underrepresented languages.. """,unsafe_allow_html=True)

	st.write(""" <center><a href="https://ibb.co/6F0kMrg"><img src="https://i.ibb.co/KWxJPbF/image.png" alt="image" border="0"></a><br />Figure 24: Google Translate news on Tamazight language support (Amazighworldnews 1, 2).<br /> </center>""",unsafe_allow_html=True)
	st.write("")
	st.subheader("Code:")
	st.write("""<b>Data Prep and EDA: </b>""",unsafe_allow_html=True)
	st.write("""https://drive.google.com/drive/folders/1QvSNVwY0pc176zVzGE9nJ_hYTw3n23py?usp=share_link """,unsafe_allow_html=True)
	st.write("""<b>Model 1:</b>""",unsafe_allow_html=True)
	st.write("""https://drive.google.com/drive/folders/17pYNb-ri2-Ho2YggqoVIw76dOe18YTZE?usp=share_link""",unsafe_allow_html=True)
	st.write("""<b>Model 2:</b>""",unsafe_allow_html=True)
	st.write("""https://drive.google.com/drive/folders/1OeqEn0OpoE9_BuYrc6O2uro6XvY-fQx0?usp=share_link""",unsafe_allow_html=True)
	st.write("""<b>Model 3:</b>""",unsafe_allow_html=True)
	st.write("""https://drive.google.com/drive/folders/183N137KlF9hHQ6YYPsM2OJfsMZyu-v0-?usp=share_link""",unsafe_allow_html=True)
	st.write("""<b>Model 4:</b>""",unsafe_allow_html=True)
	st.write("""https://drive.google.com/drive/folders/1cDVd3Dw1d-sElkzBactJ549Y_gVncdwU?usp=share_link""",unsafe_allow_html=True)


def Conclusions():
	st.header("Conclusions")
	st.write("""<a href="https://ibb.co/KxFgh1z"><img src="https://i.ibb.co/LdnyRGh/image.png" alt="image" border="0"></a><br />Figure 25: Quote from <a href ="https://www.euronews.com/2023/01/31/morocco-amazigh">EuroNews.</a><br />""",unsafe_allow_html=True)
	st.write("")
	st.write("A significant challenge in advancing Tamazight language technology lies in the limited compatibility of existing language models with the Tifinagh script. Many state-of-the-art tools fail to support this unique script, leaving a gap in their ability to accurately represent and process Tamazight in its authentic written form. This lack of script support poses a barrier not only for computational efforts but also for cultural preservation, as it limits the accessibility and visibility of Tifinagh in digital and educational platforms.")
	st.write("In addition to script limitations, the disparity in vocabulary sizes between Tamazight and English creates another obstacle. In the dataset used, Tamazight’s 7,161-word vocabulary vastly exceeds English’s 3,929 words, leading to an imbalance that can affect the training and performance of language models. This difference may result in the model underperforming in effectively translating nuanced expressions, highlighting the need for balanced and enriched datasets.")
	st.write("The issue is compounded by the limited variety within the dataset itself. A lack of diverse linguistic examples reduces the model’s ability to capture the richness of Tamazight’s multiple dialects and cultural expressions. This restricted representation undermines the potential of AI-driven tools to fully embrace the complexity and beauty of the language, making it imperative to invest in gathering broader and more inclusive linguistic data. This is clearly seen by the dataset not showing high accuracies with most of the models, even when using Google Translate, which has been proven to have high accuracies with many old languages including Tifinagh-scripted Tamazight")
	st.write("""<a href="https://ibb.co/bJYyX13"><img src="https://i.ibb.co/YZJKh2W/image.png" alt="image" border="0"></a><br />Figure 26: Amazigh Traditional Clothing (<a href = "https://www.reuters.com/world/africa/moroccos-amazigh-speakers-fear-indigenous-language-fading-2023-01-30/">Reuters </a>).<br />""",unsafe_allow_html=True)
	st.write("")
	st.write("To address these challenges, concerted efforts are required to ensure the preservation and growth of Tamazight in the digital era. Enhancing support for Tifinagh script in language technologies is a critical first step. Additionally, expanding datasets to include a wide array of dialects, contexts, and expressions can foster better generalization and more accurate translation. Collaboration between government initiatives and technological advancements can amplify these efforts, creating opportunities for Tamazight to thrive.")
	st.write("Ultimately, the revitalization of Tamazight requires a multifaceted approach that bridges linguistic preservation with cutting-edge technology. The Moroccan government’s recent commitment to increased funding for Amazigh-language initiatives demonstrates a promising direction. Coupled with the global pivot towards English and the rise of AI, this moment presents a unique opportunity to elevate Tamazight as a living, dynamic language both within Morocco and beyond. By addressing these challenges head-on, we can contribute to safeguarding a vital piece of cultural heritage while embracing the potential of technological progress.")
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
