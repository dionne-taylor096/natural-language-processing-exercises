
# Project Title

![Project Banner](path/to/banner_image.png)

*Project banner image credits: [Source Name](image_source_url)*

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Setup](#setup)
- [Data Preprocessing](#data-preprocessing)
- [Model Selection and Training](#model-selection-and-training)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

## Overview

- This project aims to classify the "100 Most Starred Github Repositories" programming languages using natural language processing (NLP) techniques. The goal is to develop a model that can accurately predict the programming language based on the content of the README files.

## Dataset

- The dataset used in this project consists of the "100 Most Starred Github Repositories" and their associated programming languages. It includes information about the repositories and their README files. The dataset contains a total of [insert number of records] records with [insert number of features] features. The target variable is the programming language. The dataset source can be found [https://api.github.com/search/repositories](insert dataset source link).

## Setup

- The following Python libraries and their versions are required for this project:

pandas 
numpy 
scikit-learn 
matplotlib 
seaborn 
nltk

- To set up the project, follow these steps:

Install Python
Create a virtual environment: python -m venv project-env.
Activate the virtual environment:
On Windows: project-env\Scripts\activate.
On macOS and Linux: source project-env/bin/activate.
Install the required libraries: pip install -r requirements.txt.
Download the dataset and save it in the project directory.

- To acquire the necessary files using the GitHub API, follow these steps:

Obtain a personal access token from GitHub:
Go to your GitHub account settings.
Navigate to "Developer settings" -> "Personal access tokens".
Click on "Generate new token" and follow the prompts to create a new token.
Make sure to select the necessary scopes and permissions for accessing the desired repositories and their contents.
Once generated, copy the access token as it will be required for authentication.
Set up the required Python libraries:
Install the requests library: pip install requests.
Create a function to retrieve repository data from the GitHub API:


## Data Preprocessing

- The data preprocessing steps include:

1. Performs preprocessing steps:

    A. Convert text to lowercase: The code applies the lower() method to each word in the 'Readme' column, converting all text to lowercase. This step helps in standardizing the text and ensuring case-insensitive matching.
    B. Remove HTML tags: The code uses the BeautifulSoup library to remove any HTML tags present in the 'Readme' column. It applies the get_text() method to extract only the text content from HTML.
    C. Remove punctuation marks: The code uses regular expressions to remove punctuation marks from the 'Readme' column. It replaces any non-word characters (excluding whitespace) with an empty string.
    D. Remove numerical digits: The code uses regular expressions to remove numerical digits from the 'Readme' column. It replaces any sequence of digits with an empty string.
    E. Remove URLs: The code uses regular expressions to remove URLs from the 'Readme' column. It replaces any URLs starting with 'http' or 'www' with an empty string.
    F. Remove special characters: The code uses regular expressions to remove any special characters from the 'Readme' column. It replaces any non-alphanumeric characters (excluding whitespace) with an empty string.
    G. Remove duplicate words: The code splits the 'Readme' column into individual words, removes duplicate words using the set() function, and then joins the unique words back into a string.


2. Applied various transformations:

    1. Defines a list of top languages, which is ['Python', 'C++', 'JavaScript']. These are the languages that will be labeled individually, while any other language will be labeled as 'Other'.
    2. The function then loops through each row of the DataFrame using the iterrows() method. For each row, it retrieves the index and the row data.
    3. Inside the loop, it checks if the value in the 'Language' column of the current row is present in the top_languages list using the in operator.
    4. If the language is in the top_languages list, it assigns the language label to the 'Language' column for that row using the at accessor. This effectively replaces the existing value in the 'Language' column with the corresponding top language.
    5. If the language is not in the top_languages list, it assigns the label 'Other' to the 'Language' column for that row.
    6. After looping through all the rows, the function returns the modified DataFrame with the updated language labels.

3. Preprocessed the text data in the DataFrame `df`. It typically involves tokenization, removing stopwords, and applying other text-specific transformations. The function modifies the `Readme` column in-place, pre-processing the text data.

    A. expand_contractions: Expanded contractions in the text using a predefined mapping of contractions to their expanded forms. It splits the text into words, checks if each word is a contraction, and replaces it with its expanded form if available.

    B. expand_abbreviations: Expanded abbreviations in the text using a predefined mapping of abbreviations to their expanded forms. It splits the text into words, checks if each word is an abbreviation, and replaces it with its expanded form if available.

    C. remove_short_or_long_words: Removed words from the text that have lengths below a minimum threshold (min_length) or above a maximum threshold (max_length). It splits the text into words and filters out words that do not meet the length criteria.

4. Applied random oversampling to address class imbalance in the DataFrame `df`. This creates a binary target variable indicating the language as Python or not, and then uses the RandomOverSampler algorithm to oversample the minority class (Python). The function returns a resampled DataFrame with balanced classes.

5. Resampled the DataFrame from the CSV file `'resampled_readme.csv'` and assigns it to the variable `df`. Resampling was emplyed due to the imbalance of target classes or languages, in the case.

6. Removed stopwords from the text data in the DataFrame `df`. Stopwords are commonly used words that do not carry significant meaning in the context of the analysis. The function modifies the `Readme` column in-place, removing stopwords from the text data.

7. Performed part-of-speech (POS) tagging and lemmatization on the text data in the DataFrame `df`. POS tagging assigns a grammatical label to each word in the text, and lemmatization reduces words to their base or root form. 

## Model Selection and Training

- Several machine learning models were considered for this project, including [insert list of models]. The model selection process involved evaluating the performance of each model using appropriate metrics. The best-performing model was selected based on [insert selection criteria]. The model was trained using [insert training process], including hyperparameter tuning and cross-validation, if applicable.

## Results

- The project results indicate that [insert best-performing model] achieved the highest accuracy of [insert accuracy] on the test set. The precision, recall, and F1-score for each programming language are as follows:

SVM Model Accuracy: 1.0

SVM Classification Report:
              precision    recall  f1-score   support

       Other       1.00      1.00      1.00        18
      Python       1.00      1.00      1.00        17

    accuracy                           1.00        35
   macro avg       1.00      1.00      1.00        35
weighted avg       1.00      1.00      1.00        35

SVM Model Accuracy: 1.0

Naive Bayes Classification Report:
              precision    recall  f1-score   support

       Other       1.00      0.94      0.97        18
      Python       0.94      1.00      0.97        17

    accuracy                           0.97        35
   macro avg       0.97      0.97      0.97        35
weighted avg       0.97      0.97      0.97        35

Naive Bayes Model Accuracy: 0.9714285714285714


Logistic Regression Classification Report:
              precision    recall  f1-score   support

       Other       1.00      1.00      1.00        18
      Python       1.00      1.00      1.00        17

    accuracy                           1.00        35
   macro avg       1.00      1.00      1.00        35
weighted avg       1.00      1.00      1.00        35

Logistic Regression Model Accuracy: 1.0
## Future Work

- For future work, the following improvements and extensions can be considered:

Explore advanced NLP techniques, such as word embeddings or deep learning models, to further improve classification performance.
Collect additional features or data sources that can provide more information about the repositories and improve the model's predictive power.
Investigate the impact

## Acknowledgements

- List any references, articles, or resources used during the project.
- Acknowledge any collaborators or external support, if applicable.

