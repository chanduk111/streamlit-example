import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

st.title('Twitter Sentiment Analysis')
st.write('Welcome to my sentiment analysis app!')

# Create a form for user input
form = st.form(key='sentiment-form')
user_input = form.text_area('Enter your text')
submit = form.form_submit_button('Submit')

# Initialize the sentiment analysis tool
analyzer = SentimentIntensityAnalyzer()

if submit:
    # Analyze sentiment
    sentiment_scores = analyzer.polarity_scores(user_input)
    
    # Determine the sentiment category
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        sentiment_category = 'Positive'
    elif compound_score <= -0.05:
        sentiment_category = 'Negative'
    else:
        sentiment_category = 'Neutral'
    
    # Display sentiment analysis results
    st.write("Sentiment Analysis Results:")
    st.write(f"Sentiment Category: {sentiment_category}")
    st.write(f"Positive: {sentiment_scores['pos']:.2f}")
    st.write(f"Negative: {sentiment_scores['neg']:.2f}")
    st.write(f"Neutral: {sentiment_scores['neu']:.2f}")
