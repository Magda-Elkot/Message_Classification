# E-Mail Classification NLP
    This project aims to classify emails into spam or not spam using Natural Language Processing (NLP) techniques. 
    The project uses a dataset containing SMS messages that have already been classified as spam or not spam.

# The NLP techniques used in this project include:
    ## Text preprocessing:
        Convert all characters to lowercase, remove stop words, punctuation, and apply stemming. 
        This technique is used to clean the text data and reduce the number of unique words in the dataset.
    ## CountVectorizer:
         This technique is used to convert preprocessed text data into numerical feature vectors. 
         CountVectorizer counts the frequency of each word in the text data and creates a feature vector for each document.
    ## TfidfVectorizer:
        This technique is used to convert preprocessed text data into numerical feature vectors. 
        TfidfVectorizer calculates the term frequency-inverse document frequency (TF-IDF) of each word in the text data and creates 
        a feature vector for each document.
    ## Support Vector Machine (SVM):
        SVM is a machine learning algorithm used for classification. 
        SVM is trained on the feature vectors created by CountVectorizer and TfidfVectorizer to classify emails into spam or not spam.
    ## Random Forest (RF): 
        RF is a machine learning algorithm used for classification. 
        RF is trained on the feature vectors created by CountVectorizer and TfidfVectorizer to classify emails into spam or not spam.
    ## Accuracy, Precision, and Recall metrics: 
        These metrics are used to evaluate the performance of the classifiers. 
        Accuracy measures the percentage of correctly classified emails. 
        Precision measures the percentage of correctly classified spam emails out of all classified spam emails. 
        Recall measures the percentage of correctly classified spam emails out of all actual spam emails.    

        ![image](https://github.com/Magda-Elkot/Message_Classification/assets/121414067/48ccddf4-f4a0-4fa5-aa12-a9c9325f8170)


# Installation
    The following libraries need to be installed to run this project:

      numpy
      pandas
      matplotlib
      nltk
      scikit-learn
    To install the libraries, use the following command in the terminal:
      `pip install numpy pandas matplotlib nltk scikit-learn`
    Also, download the 'stopwords' corpus from nltk library using the following command in the 
    Python console:
      import nltk
      nltk.download('stopwords')

# Usage
    The project uses the following steps for email classification:
      1. Load the dataset from CSV files using pandas.
      2. Remove unnecessary columns from the dataset.
      3. Preprocess the text data by converting all characters to lowercase, removing stop words, punctuations, and applying stemming.
      4. Convert the preprocessed text data into numerical feature vectors using CountVectorizer and TfidfVectorizer.
      5. Split the dataset into training and testing sets.
      6. Train a Support Vector Machine (SVM) and Random Forest (RF) classifiers on both the 
      CountVectorizer and TfidfVectorizer feature vectors.
      7. Evaluate the performance of each classifier using accuracy, precision, and recall metrics.
      8. Visualize the performance of each classifier using bar plots.
      
    To run the project, open the Python console and execute the script. The script will load the dataset, preprocess the text data, train the classifiers, 
    evaluate their performance, and visualize the results.
