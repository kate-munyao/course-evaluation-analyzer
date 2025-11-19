# Lab Submission Instructions

---

## Student Details

**Name of the team on GitHub Classroom:**

**Team Member Contributions:**

**Member 1**

| **Details**                                                                                        | **Comment** |
|:---------------------------------------------------------------------------------------------------|:------------|
| **Student ID:**                                                                                    |    144954         |
| **Name:**                                                                                          |Brianna Muthoni        |
| **What part of the lab did you personally contribute to,** <br>**and what did you learn from it?** |   Worked on improving the Streamlit interface, including sample input buttons, layout adjustments, and connecting the sentiment and topic model predictions to the app. Helped debug session state issues, chart sizes, and the input field logic. Also contributed to testing the model outputs especially checking neutral vs negative prediction consistency  and verifying that all saved model files loaded correctly. Learned about Streamlit state management, UI–model integration, and how to troubleshoot deployment errors.     |

**Member 2**

| **Details**                                                                                        | **Comment** |
|:---------------------------------------------------------------------------------------------------|:------------|
| **Student ID:**                                                                                    |146569         |
| **Name:**                                                                                          | Jude Muriithi         |
| **What part of the lab did you personally contribute to,** <br>**and what did you learn from it?** |   Worked on integrating topic predictions and sentiment predictions into the final combined dataset. Built the summary statistics, confusion matrix visualizations, and word cloud generation. Helped write logic to map model outputs into readable insights used in the UI. Learned about creating meaningful model visualizations, working with Matplotlib/Seaborn, and how to convert raw ML outputs into user-friendly dashboards for non-technical audiences.          |

**Member 3**

| **Details**                                                                                        | **Comment** |
|:---------------------------------------------------------------------------------------------------|:------------|
| **Student ID:**                                                                                    | 147700            |
| **Name:**                                                                                          |    Natasha George         |
| **What part of the lab did you personally contribute to,** <br>**and what did you learn from it?** |    Worked on cleaning and preprocessing the dataset, including text normalization, stopword removal, and tokenization. Contributed to testing multiple preprocessing strategies to reduce noise before topic modelling. Helped validate that the final cleaned text produced stable topic assignments and improved sentiment predictions. Learned about the impact of preprocessing choices on model performance, how to debug inconsistent text outputs across notebooks, and how data quality directly affects downstream results.        |

**Member 4**

| **Details**                                                                                        | **Comment** |
|:---------------------------------------------------------------------------------------------------|:------------|
| **Student ID:**                                                                                    | 145656            |
| **Name:**                                                                                          |  Ronald Mutarura          |
| **What part of the lab did you personally contribute to,** <br>**and what did you learn from it?** | Developed the sentiment analysis model, including TF-IDF feature extraction, model selection, and hyperparameter tuning. Ran cross-validation for Logistic Regression, Naïve Bayes, Decision Trees, and Random Forests to determine the best-performing classifier. Helped evaluate precision, recall, and F1-score for each class. Learned about balancing multi-class sentiment data (positive, neutral, negative), the importance of evaluation metrics beyond accuracy, and the challenges of interpreting neutral sentiment.            |

**Member 5**

| **Details**                                                                                        | **Comment** |
|:---------------------------------------------------------------------------------------------------|:------------|
| **Student ID:**                                                                                    |145060             |
| **Name:**                                                                                          |Catherine Munyao           |
| **What part of the lab did you personally contribute to,** <br>**and what did you learn from it?** | Implemented the topic modelling pipeline using LDA, tuned the number of topics, and evaluated coherence scores. Worked on creating interpretable topic labels based on top keywords. Assisted in connecting the topic model outputs to the combined dataset used for sentiment analysis. Learned how unsupervised learning differs from supervised approaches, how to interpret topic-word distributions, and the importance of naming topics in a way that stakeholders can understand.            |

## Scenario

Your client, a university, is seeking to enhance their qualitative analysis of
student course evaluations collected from students. They have provided you
with a dataset containing student course evaluation for two courses in the
Business Intelligence Option. The two courses are:
- BBT 4106: Business Intelligence I
- BBT 4206: Business Intelligence II

The client wants you to use Natural Language Processing (NLP) techniques to identify
the key topics (themes) discussed in the course evaluations. They would also like to
get the sentiments (positive, negative, neutral) of each theme in the course evaluation.

Lastly, the client would like an interface through which they can provide input in the
form of new textual data (one student's textual evaluation at a time) and the output
expected is:
1. The topic (theme) that the new textual data is talking about.
2. The sentiment (positive, negative, neutral) of the new textual data.

Use one of the following to create a demo interface for your client:
- Hugging Face Spaces using a Gradio App – [https://huggingface.co/spaces](https://huggingface.co/spaces)
- Streamlit Community Cloud (Streamlit Sharing) using a Streamlit App – [https://share.streamlit.io](https://share.streamlit.io)
---
## Dataset

Use the course evaluation dataset provided in Google Classroom.

## Interpretation and Recommendation

Provide a brief interpretation of the results and a recommendation for the client.
- Interpret what the discovered topics mean and why certain sentiments dominate
- Provide recommendations based on your results. **Do not** recommend anything that is not supported by your results.

## Video Demonstration

Submit the link to a short video (not more than 4 minutes) demonstrating the topic modelling and the sentiment analysis.
Also include (in the same video) the user interface hosted on hugging face or streamlit.

| **Key**                             | **Value** |
|:------------------------------------|:----------|
| **Link to the video:**              |     https://youtu.be/vIFKjn64KNg      |
| **Link to the hosted application:** |  https://course-evaluation-analyzer-uemqsxoe3yg9v7eajcwt5u.streamlit.app/         |


## Grading Approach

| Component                            | Weight | Description                                                       |
|:-------------------------------------|:-------|:------------------------------------------------------------------|
| **Data Preprocessing & Analysis**    | 20%    | Cleaning, preprocessing, and justification of chosen methods.     |
| **Topic Modelling**                  | 20%    | Correctness, interpretability, and coherence of topics.           |
| **Sentiment Analysis**               | 20%    | Appropriate model choice and quality of sentiment classification. |
| **Interface Design & Functionality** | 20%    | Usability, interactivity, and deployment success.                 |
| **Interpretation & Recommendation**  | 10%    | Logical, evidence-based, and actionable insights.                 |
| **Presentation (Video & Clarity)**   | 10%    | Clarity, professionalism, and demonstration of understanding.     |
