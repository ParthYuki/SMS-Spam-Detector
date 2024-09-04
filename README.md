# SMS Spam Classifier
This project is an SMS Spam Classifier that uses Natural Language Processing (NLP) and machine learning techniques to classify SMS messages as either "spam" or "ham" (not spam). The classification is achieved through the implementation of various machine learning models, with the final prediction being made using an ensemble voting classifier of the top three models.

## Table of Contents
• Overview  
• Dataset  
• Installation  
• Usage  
• Project Structure  
• Methodology  
• Results  
• Contributing  
## Overview
The primary goal of this project is to detect spam messages in SMS data. The project involves:  
1. Data Cleaning and Preprocessing: Using NLP techniques to clean and preprocess the text data.  
2. Model Training: Applying various machine learning classification models.  
3. Model Selection: Choosing the top three models based on their accuracy.  
4. Ensemble Method: Implementing a voting classifier to combine the top three models and improve prediction performance.  
## Dataset
The dataset used in this project is the SMS Spam Collection Dataset available on Kaggle. It consists of a collection of 5,574 SMS messages labeled as either "spam" or "ham."

## Installation
To run this project, you'll need Python installed on your machine along with the following libraries:

pip install numpy pandas scikit-learn nltk
## Usage

1. Clone this repository:
git clone https://github.com/ParthYuki/SMS-Spam-Detector.git  
2. Navigate to the project directory:
cd sms-spam-classifier  
3. Run the Jupyter notebook to train the models and classify messages:
jupyter notebook sms-spam-classifier.ipynb
## Project Structure
• sms-spam-classifier.ipynb: The main Jupyter notebook containing the code for data preprocessing, model training, and evaluation.   
• spam.csv: The dataset file containing SMS messages and their labels.  
• README.md: This readme file providing an overview and instructions for the project.  
## Methodology
1. Data Preprocessing:  
• Load and inspect the dataset.    
• Clean the data by removing unnecessary columns, handling missing values, and preprocessing the text (e.g., removing stopwords, tokenization). 

2. Model Training:  
Several machine learning models are trained on the processed data, including:  
• Logistic Regression  
• Support Vector Machines (SVM)  
• Naive Bayes  
• Random Forest  
• K-Nearest Neighbors (KNN)  
Each model's performance is evaluated using accuracy as the metric.

3. Model Selection:  
• The top three models with the highest accuracy are selected.

4. Ensemble Voting Classifier:  
• A voting classifier is created by combining the selected models to improve overall performance and robustness.
## Results
The voting classifier, which combines the best-performing models, provides the final classification for each SMS message, achieving a high accuracy in distinguishing between spam and ham messages.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
