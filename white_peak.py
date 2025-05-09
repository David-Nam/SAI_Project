import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import re
import os

# Load the model and tokenizer for Korean sentiment analysis
model_name = "WhitePeak/bert-base-cased-Korean-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to clean KakaoTalk messages
def clean_kakao_message(text):
    # Remove timestamps, system messages, etc.
    if isinstance(text, str):
        # Remove photo, video attachments
        text = re.sub(r'\[사진\]|\[이모티콘\]|\[동영상\]|\[파일\]', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove other non-text content indicators
        text = re.sub(r'\[샵검색\]|\[지도\]', '', text)
        return text.strip()
    return ""

# Function to analyze sentiment of a text
def analyze_sentiment(text, model, tokenizer):
    if not text or len(text.strip()) < 2:  # Skip empty or very short texts
        return None
    
    # Encode the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
        probabilities = torch.nn.functional.softmax(predictions, dim=1)
        
        # Get the predicted class (0 = negative, 1 = positive)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
    # Return sentiment (negative or positive) and confidence score
    sentiment = "Positive" if predicted_class == 1 else "Negative"
    return {"sentiment": sentiment, "confidence": confidence, "text": text}

# Load the KakaoTalk CSV file
def analyze_kakao_csv(csv_path):
    try:
        # Try to detect encoding
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='cp949')
        except:
            print("Please check CSV file encoding. Typically 'utf-8' or 'cp949' is used.")
            return
    
    # Detect the structure of the CSV
    # Assuming the CSV has columns for date, sender, and message
    # Adjust column names based on your actual CSV structure
    print("CSV file structure:", df.columns.tolist())
    
    # Try to identify message column
    message_col = None
    if '내용' in df.columns:
        message_col = '내용'
    elif 'message' in df.columns or 'Message' in df.columns:
        message_col = 'message' if 'message' in df.columns else 'Message'
    else:
        # Try to guess which column contains the message content
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].str.len().mean() > 10:
                message_col = col
                break
    
    if not message_col:
        print("Could not find a column containing message content.")
        return
    
    # Clean messages
    df['cleaned_message'] = df[message_col].apply(clean_kakao_message)
    
    # Analyze sentiment for each message
    results = []
    
    print(f"Analyzing a total of {len(df)} messages...")
    for message in tqdm(df['cleaned_message']):
        sentiment_result = analyze_sentiment(message, model, tokenizer)
        if sentiment_result:
            results.append(sentiment_result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

# Visualize the sentiment analysis results
def visualize_results(results_df):
    if results_df is None or len(results_df) == 0:
        print("No results to analyze.")
        return
    
    # Count of positive vs negative messages
    sentiment_counts = results_df['sentiment'].value_counts()
    
    # Plot sentiment distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='sentiment', data=results_df, palette=['red', 'green'])
    plt.title('KakaoTalk Chat Sentiment Analysis Results')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Messages')
    
    # Add percentage labels
    total = len(results_df)
    for i, count in enumerate(sentiment_counts):
        percentage = count / total * 100
        plt.text(i, count + 5, f'{percentage:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig('sentiment_analysis_results.png')
    plt.show()
    
    # Display average confidence by sentiment
    avg_confidence = results_df.groupby('sentiment')['confidence'].mean()
    print("\nAverage confidence by sentiment:")
    for sentiment, conf in avg_confidence.items():
        print(f"{sentiment}: {conf:.4f}")
    
    # Display sample messages with their sentiment
    print("\nSample message analysis results:")
    for sentiment in results_df['sentiment'].unique():
        sample = results_df[results_df['sentiment'] == sentiment].sort_values('confidence', ascending=False).head(3)
        print(f"\n{sentiment} message examples (sorted by confidence descending):")
        for i, row in sample.iterrows():
            print(f"- {row['text']} (confidence: {row['confidence']:.4f})")

# Main function
def main():
    csv_path = input("Enter the path to the KakaoTalk chat CSV file: ")
    
    if not os.path.exists(csv_path):
        print("File not found. Please check the path and try again.")
        return
    
    results_df = analyze_kakao_csv(csv_path)
    
    if results_df is not None:
        # Save results to CSV
        results_df.to_csv('sentiment_analysis_results.csv', index=False, encoding='utf-8-sig')
        print("Analysis results have been saved to sentiment_analysis_results.csv.")
        
        # Visualize results
        visualize_results(results_df)

if __name__ == "__main__":
    main()