# Twitter Sentiment Analysis on Climate Change

This project analyzes Twitter sentiment regarding climate change using machine learning and natural language processing techniques. It utilizes a Multinomial Naive Bayes model and the VADER lexicon for sentiment classification.

## Overview

The project consists of the following steps:

1.  Data Preprocessing: Cleaning and preparing tweet text data.
2.  Model Training: Training a sentiment classification model using labeled tweet data.
3.  Twitter Data Collection: Gathering recent tweets related to climate change.
4.  Sentiment Analysis: Applying the trained model and VADER lexicon to classify the collected tweets.
5.  Visualization and Reporting: Displaying sentiment distributions and summarizing key findings.

## Installation

--To run this project, you'll need to install the following Python libraries:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn tweepy

--You will also need to download the following NLTK resources:

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')


--Twitter API Setup

You need a Twitter developer account and API keys to gather data. Make sure to do this before starting. Update the following placeholders in the code with your actual API credentials:

API_KEY = 'YOUR_API_KEY'
API_SECRET_KEY = 'YOUR_API_SECRET_KEY'
ACCESS_TOKEN = 'YOUR_ACCESS_TOKEN'
ACCESS_TOKEN_SECRET = 'YOUR_ACCESS_TOKEN_SECRET'


--Dataset

The project utilizes the Sentiment140 dataset, which is a large dataset of tweets labeled with sentiment. You can download the dataset from the following link:

Sentiment140 Dataset: https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download

Specifically, the file training.1600000.processed.noemoticon.csv should be in the same directory as your python script.

--Usage

1.Clone this repository to your local machine.

2.Install the required libraries using the commands above.

3.Download the Sentiment140 dataset and place training.1600000.processed.noemoticon.csv in your project directory

4.Add your Twitter API keys.

5.Run the Python script to perform sentiment analysis:

python your_script_name.py

Replace your_script_name.py with the name of your Python file.

Project Structure
├── README.md         
├── your_script_name.py    # Main script
└── training.1600000.processed.noemoticon.csv # Dataset

--Contributing

Feel free to fork the repository and submit pull requests with your contributions.

--License

This project is licensed under the MIT License. (If you want to add a license).

Explanation of key parts:

*   Installation: Clearly lists the Python packages that need to be installed. It also includes the nltk download commands which are crucial for the code to run.
*   Twitter API Setup:  Highlights the need for Twitter API keys and provides guidance on where to input them.
*   Dataset: Provides a direct link to the Sentiment140 dataset and clarifies which specific file is needed.
*   Usage: Explains how to run the project, assuming the user has the necessary files and the twitter api keys
*   Project Structure: Gives a simple view of how the files should be organized.
*   Contributing/License: Standard sections to encourage participation and specify usage.

To use this README:

1.  Save it as `README.md` in the root of your project repository.
2.  Replace placeholders like `your_script_name.py`, your API keys and the license if you are including one.
3.  The GitHub platform will automatically render it.
