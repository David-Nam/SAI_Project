import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import re
import os
import json
from datetime import datetime

# Set font for plots (macOS 한글 지원)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# Function to clean KakaoTalk messages
def clean_kakao_message(text):
    # Remove timestamps, system messages, etc.
    if isinstance(text, str):
        # Remove photo, video attachments
        text = re.sub(r'\[Photo\]|\[Emoticon\]|\[Video\]|\[File\]', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove other non-text content indicators
        text = re.sub(r'\[Shop\]|\[Map\]', '', text)
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
        
        # Get the predicted class and confidence
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Map class index to label using the model's id2label mapping
        sentiment = model.config.id2label[predicted_class]
    
    return {"sentiment": sentiment, "confidence": confidence, "text": text}

# Load the KakaoTalk CSV file
def analyze_kakao_csv(csv_path, model, tokenizer):
    try:
        # Try to detect encoding
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='cp949')
        except:
            print("Please check the CSV file encoding. Usually 'utf-8' or 'cp949'.")
            return
    
    # Detect the structure of the CSV
    print("CSV file structure:", df.columns.tolist())
    
    # Try to identify message column and timestamp column
    message_col = None
    timestamp_col = None
    possible_cols = ['Text', 'Message', 'Content', 'text', 'message', 'content']
    possible_time_cols = ['Date', 'Time', 'Timestamp', 'date', 'time', 'timestamp']
    
    for col in possible_cols:
        if col in df.columns:
            message_col = col
            break
    
    for col in possible_time_cols:
        if col in df.columns:
            timestamp_col = col
            break
    
    if not message_col:
        # Try to guess which column contains the message content
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].str.len().mean() > 10:
                message_col = col
                break
    
    if not message_col:
        print("Could not find a column containing message content.")
        return
    
    print(f"Using '{message_col}' as the message column.")
    if timestamp_col:
        print(f"Using '{timestamp_col}' as the timestamp column.")
    
    # Clean messages
    df['cleaned_message'] = df[message_col].apply(clean_kakao_message)
    
    # Analyze sentiment for each message
    results = []
    
    print(f"Analyzing {len(df)} messages...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        message = row['cleaned_message']
        if not message or len(message.strip()) < 2:  # Skip empty or very short texts
            continue
            
        # Get timestamp if available
        timestamp = row[timestamp_col] if timestamp_col else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Use the pipeline to analyze sentiment
        sentiment_result = analyze_sentiment(message, model, tokenizer)
        if sentiment_result:
            sentiment_result['timestamp'] = timestamp
            results.append(sentiment_result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

# Visualize the sentiment analysis results
def visualize_results(results_df, model_name="KCElectra"):
    if results_df is None or len(results_df) == 0:
        print("No results to analyze.")
        return
    
    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Count each class and draw bar chart with dynamic colors
    sentiment_counts = results_df['sentiment'].value_counts()
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("husl", len(sentiment_counts))
    ax = sentiment_counts.plot(kind='bar', color=palette)
    plt.title(f'Sentiment Analysis Results: {model_name}')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Messages')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_sentiment_analysis_{timestamp}.png')
    plt.show()
    
    # Save results to CSV
    results_df.to_csv(f'{output_dir}/{model_name}_results_{timestamp}.csv', index=False, encoding='utf-8-sig')
    
    # Create JSON output
    json_output = {
        "model_name": model_name,
        "analysis_timestamp": timestamp,
        "total_messages": len(results_df),
        "sentiment_distribution": sentiment_counts.to_dict(),
        "average_confidence": results_df.groupby('sentiment')['confidence'].mean().to_dict(),
        "messages": [
            {
                "text": row['text'],
                "sentiment": row['sentiment'],
                "confidence": float(row['confidence']),
                "timestamp": row['timestamp']
            }
            for _, row in results_df.iterrows()
        ]
    }
    
    # Save JSON output
    json_path = f'{output_dir}/{model_name}_results_{timestamp}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    
    # Display average confidence by sentiment
    print("\nAverage confidence by sentiment:")
    avg_confidence = results_df.groupby('sentiment')['confidence'].mean().reset_index()
    for _, row in avg_confidence.iterrows():
        print(f"{row['sentiment']}: {row['confidence']:.4f}")
    
    # Display sample messages with their sentiment
    print("\nSample message analysis results:")
    for sentiment in results_df['sentiment'].unique():
        sample = results_df[results_df['sentiment'] == sentiment].sort_values('confidence', ascending=False).head(3)
        print(f"\n{sentiment} message examples (highest confidence):")
        for _, row in sample.iterrows():
            print(f"- [{row['timestamp']}] {row['text'][:50]}{'...' if len(row['text']) > 50 else ''} (Confidence: {row['confidence']:.4f})")
    
    print(f"\nResults have been saved to:")
    print(f"- CSV: {output_dir}/{model_name}_results_{timestamp}.csv")
    print(f"- JSON: {output_dir}/{model_name}_results_{timestamp}.json")
    print(f"- Plot: {output_dir}/{model_name}_sentiment_analysis_{timestamp}.png")

# Main function
def main():
    print("nlp04/kcelectra Model Analysis")
    print("-----------------------------")
    
    # Load the model and tokenizer
    print("Loading model...")
    model_name = "nlp04/korean_sentiment_analysis_kcelectra"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Get CSV file path from user
    csv_path = input("Enter the path to your KakaoTalk CSV file: ")
    
    if not os.path.exists(csv_path):
        print("File not found. Please check the path.")
        return
    
    # Analyze the CSV file
    results_df = analyze_kakao_csv(csv_path, model, tokenizer)

    if results_df is not None:
        # Visualize the results
        print("\nVisualizing results...")
        visualize_results(results_df, "KCElectra")
        print(f"\nAnalysis complete! Results saved to the 'results' folder.")


if __name__ == "__main__":
    main()