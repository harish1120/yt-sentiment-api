import matplotlib
import matplotlib.colors as mcolors
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse

import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle
from schema import CommentItem, commentData, commentwithTSData, sentimentCount, SentimentArray, comments

from fastapi.middleware.cors import CORSMiddleware

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

def load_model(model_path, vectorizer_path):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
      
        return model, vectorizer
    except Exception as e:
        raise


# Initialize the model and vectorizer
model, vectorizer = load_model("/Users/harishsundaralingam/myworkspace/sentiment_analysis/lgbm_model.pkl", "/Users/harishsundaralingam/myworkspace/sentiment_analysis/tfidf_vectorizer.pkl")  

# Initialize the model and vectorizer
# model, vectorizer = load_model_and_vectorizer("my_model", "1", "./tfidf_vectorizer.pkl")  # Update paths and versions as needed
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "chrome-extension://kbanbmpkagefokgdabbhcdkmlfgdhbgj"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def home():
    return {"message": "Sentiment Analysis API is running!"}

@app.post ('/predict', status_code=status.HTTP_202_ACCEPTED)
async def predict(data: commentData):
    comments = data.comments

    if not comments:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No comments provided")
    try:
       preprocessed_comments = [preprocess_comment(comment) for comment in comments]
       transformed_comments = vectorizer.transform(preprocessed_comments)
       dense_comments = transformed_comments.toarray()
       predictions = model.predict(dense_comments).tolist()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {str(e)}")
    
    # Return the response with original comments, predicted sentiments, and timestamps
    response = [
        {
            "comment": comment,
            "sentiment": sentiment
        }
        for comment, sentiment in zip(comments, predictions)
    ]

    return response

@app.post('/predict_with_timestamps', status_code=status.HTTP_202_ACCEPTED)
def predict_with_timestamps(data: commentwithTSData):
    comments_data = data.comments

    try:
        comments = [item.text for item in comments_data]
        timestamps = [item.timestamp for item in comments_data]

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # Convert the sparse matrix to dense format
        dense_comments = transformed_comments.toarray()  # Convert to dense array
        
        # Make predictions
        predictions = model.predict(dense_comments).tolist()  # Convert to list
        
        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {str(e)}")
    
    # Return the response with original comments, predicted sentiments, and timestamps
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return response


@app.post('/generate_chart')
async def generate_chart(sentiment_data: sentimentCount):
    # Fix: Access sentiment_counts instead of count
    sentiment_count = sentiment_data.sentiment_counts

    if not sentiment_count:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No sentiment data provided")
    
    labels = ['Positive', 'Neutral', 'Negative']
    
    # Fix: Handle string keys properly
    sizes = [
        int(sentiment_count.get('1', 0)),
        int(sentiment_count.get('0', 0)),
        int(sentiment_count.get('-1', 0))
    ]
    
    try:
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        print(f"Chart sizes: {sizes}")  # Debug line
        
        # Fix: Use correct color import
        colors = [
            mcolors.to_hex('darkgreen'), 
            mcolors.to_hex('orange'), 
            mcolors.to_hex('red')
        ]

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, 
                labels=labels, 
                colors=colors, 
                autopct='%1.1f%%',
                startangle=140,
                textprops={'fontsize': 12, 'color': 'white'})
        plt.axis('equal')
        
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        return StreamingResponse(
            img_io,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=chart.png"}
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to generate chart: {str(e)}")

@app.post('/generate_wordcloud')
async def generate_wordcloud(data: comments):
    try:
        comments_data = data.comments
        
        # Fix: Access data.comments instead of just data
        preprocessed_comments = [preprocess_comment(comment) for comment in comments_data]
        
        # Debug: Check preprocessing results
        print(f"Preprocessed comments count: {len(preprocessed_comments)}")
        print(f"Sample preprocessed text: {preprocessed_comments[:3]}")
        
        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)
        
        if not text.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No text available for word cloud")
        
        print(f"Total text length: {len(text)}")
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False,
            max_words=100
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        return StreamingResponse(
            img_io,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=wordcloud.png"}
        ) 
    except Exception as e:
        print(f"Word cloud generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to generate word cloud: {str(e)}"
        )

@app.post('/generate_trend_graph')
def generate_trend_graph(data: dict):
    try:
        sentiment_data = data.get('sentiment_data', [])
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        return StreamingResponse(
            img_io,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=trend.png"}
        )
    
    except:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate chart")