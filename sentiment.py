import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')
st.title('Tool for analyzing document reviews: sentimental analysis')
st.write('You can analyze sentiment of your content here!!')

# Create a file uploader component
uploaded_file = st.file_uploader('Upload a text file', type=['txt'])

# Initialize the sentiment analysis tool
analyzer = SentimentIntensityAnalyzer()

if uploaded_file:
    # Read and analyze sentiment for the content of the uploaded file
    file_contents = uploaded_file.read().decode('utf-8')
    sentiment_scores = analyzer.polarity_scores(file_contents)
    
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
