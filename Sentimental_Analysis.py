import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import tweepy
from nltk.sentiment import SentimentIntensityAnalyzer

# Download nltk resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# ------------------------------------
# --- Step 1: Setup and Data Loading ---
# ------------------------------------

# Load Sentiment140 Dataset
# Assuming you have a CSV file named 'training.1600000.processed.noemoticon.csv' in the same directory.
# The dataset has no header by default
df_train = pd.read_csv('training.1600000.processed.noemoticon.csv',
                        encoding="ISO-8859-1",
                        names=["sentiment", "id", "date", "query", "user", "text"])

print("Training data shape:", df_train.shape)
print(df_train.head())

# ------------------------------------
# --- Step 2: Data Preprocessing ---
# ------------------------------------

# Keep only the text and label
df_train = df_train[['sentiment', 'text']]

# Convert sentiment labels
df_train['sentiment'] = df_train['sentiment'].replace(4, 1)


# Text Cleaning Function
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Removing non-alphabetical chars
    return text

df_train['cleaned_text'] = df_train['text'].apply(clean_text)
print(df_train.head())


# ------------------------------------
# --- Step 3: Machine Learning Data Prep ---
# ------------------------------------
# Data splitting for machine learning
X = df_train['cleaned_text']
y = df_train['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Training data shape: {X_train.shape}')
print(f'Test data shape: {X_test.shape}')

# ------------------------------------
# --- Step 4: Model Training ---
# ------------------------------------
# Model training with sklearn Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Detailed classification report
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# ------------------------------------
# --- Step 5: Twitter Data Gathering ---
# ------------------------------------

# Twitter API Credentials
# Replace with your own keys
API_KEY = 'F7ACvDuzo8PkOtemTivQJ54vs'
API_SECRET_KEY = 'rybdcQ2TthvIqoGIKMRLiIUeBMrjI9HL9EkR61U0UgpdcvMgLY'
ACCESS_TOKEN = '1378654558361112576-urHd6zxymdMdEyqd593l0Vrl5mIWPR'
ACCESS_TOKEN_SECRET = 'rSh1BI1zRzMafuJqIJcWQ3E9tIHoGNLBh1SkqvzgQpTY0'

# Authentication
auth = tweepy.OAuth1UserHandler(API_KEY, API_SECRET_KEY, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# Fetch Climate Tweets
def fetch_tweets(query, count=500):
    tweets = api.search_tweets(q=query, count=count, lang="en", tweet_mode='extended')
    tweet_data = [tweet.full_text for tweet in tweets]
    return tweet_data

climate_tweets = fetch_tweets(query="#ClimateChange OR #GlobalWarming", count=500)
print(f"Number of Climate Tweets Collected: {len(climate_tweets)}")

# ------------------------------------
# --- Step 6: Climate Tweet Analysis ---
# ------------------------------------
# Clean Climate Tweets
cleaned_climate_tweets = [clean_text(tweet) for tweet in climate_tweets]

# Make Predictions
predicted_sentiments = pipeline.predict(cleaned_climate_tweets)

# VADER Analysis
sia = SentimentIntensityAnalyzer()
def get_vader_sentiment_scores(text):
    return sia.polarity_scores(text)

vader_scores = [get_vader_sentiment_scores(tweet) for tweet in cleaned_climate_tweets]

print(f"Vader Scores Sample {vader_scores[:5]}")


# ------------------------------------
# --- Step 7: Aggregate and Visualize ---
# ------------------------------------
# Create a DataFrame
df_climate = pd.DataFrame({
    'text': climate_tweets,
    'cleaned_text': cleaned_climate_tweets,
    'predicted_sentiment': predicted_sentiments,
    'vader_scores': vader_scores,
})

df_climate['vader_compound'] = df_climate['vader_scores'].apply(lambda x: x['compound'])

# Classify vader sentiments
df_climate['vader_sentiment'] = df_climate['vader_compound'].apply(lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))

# Plotting Model Predictions
plt.figure(figsize=(10, 5))
df_climate['predicted_sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution of Climate Change Tweets (Model Prediction)')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.show()

# Plotting VADER scores
plt.figure(figsize=(10, 5))
df_climate['vader_sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution of Climate Change Tweets (VADER)')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.show()



# ------------------------------------
# --- Step 8: Insights and Report ---
# ------------------------------------
# Basic insights - you can expand this with more sophisticated analysis
print("--- Summary Insights ---")
model_sentiment_counts = df_climate['predicted_sentiment'].value_counts()
print("Sentiment count using model prediction:\n", model_sentiment_counts)

vader_sentiment_counts = df_climate['vader_sentiment'].value_counts()
print("Sentiment count using VADER score:\n", vader_sentiment_counts)

print("\n-- Top Positive Tweets (VADER):")
print(df_climate.sort_values(by='vader_compound', ascending=False)[['text', 'vader_compound']].head(5))

print("\n-- Top Negative Tweets (VADER):")
print(df_climate.sort_values(by='vader_compound', ascending=True)[['text', 'vader_compound']].head(5))
