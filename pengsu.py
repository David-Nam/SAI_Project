import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from tqdm import tqdm
import re
import os

# Set font for plots
plt.rcParams['font.family'] = 'Arial'
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

# Load the KakaoTalk CSV file
def analyze_kakao_csv(csv_path, sentiment_pipeline):
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
    
    # Try to identify message column
    message_col = None
    possible_cols = ['Text', 'Message', 'Content', 'text', 'message', 'content']
    
    for col in possible_cols:
        if col in df.columns:
            message_col = col
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
    
    # Clean messages
    df['cleaned_message'] = df[message_col].apply(clean_kakao_message)
    
    # Analyze sentiment for each message
    results = []
    
    print(f"Analyzing {len(df)} messages...")
    for message in tqdm(df['cleaned_message']):
        if not message or len(message.strip()) < 2:  # Skip empty or very short texts
            continue
            
        # Use the pipeline to analyze sentiment
        prediction = sentiment_pipeline(message)
        
        if prediction and len(prediction) > 0:
            result = {
                "text": message,
                "sentiment": prediction[0]["label"],
                "confidence": prediction[0]["score"]
            }
            results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

# Visualize the sentiment analysis results
def visualize_results(results_df, model_name="MLB_Care"):
    if results_df is None or len(results_df) == 0:
        print("No results to analyze.")
        return
    
    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Count of sentiments
    sentiment_counts = results_df['sentiment'].value_counts()
    
    # Use a colorful palette for multiple emotions
    palette = sns.color_palette("husl", len(sentiment_counts))
    
    # Plot sentiment distribution
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(x='sentiment', data=results_df, palette=palette)
    plt.title(f'Emotion Analysis Results: {model_name}')
    plt.xlabel('Emotion')
    plt.ylabel('Number of Messages')
    
    # Add percentage labels
    total = len(results_df)
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        if height > 0:
            percentage = height / total * 100
            ax.text(p.get_x() + p.get_width()/2., height + 5, 
                    f'{percentage:.1f}%', 
                    ha="center", fontsize=9)
    
    # Rotate x-axis labels if there are many categories
    if len(sentiment_counts) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_emotion_analysis.png')
    plt.close()
    
    # Save results to CSV
    results_df.to_csv(f'{output_dir}/{model_name}_results.csv', index=False, encoding='utf-8-sig')
    
    # Display average confidence by sentiment
    print("\nAverage confidence by emotion:")
    avg_confidence = results_df.groupby('sentiment')['confidence'].mean().reset_index()
    for _, row in avg_confidence.iterrows():
        print(f"{row['sentiment']}: {row['confidence']:.4f}")
    
    # Display sample messages with their sentiment
    print("\nSample message analysis results:")
    for sentiment in results_df['sentiment'].unique():
        sample = results_df[results_df['sentiment'] == sentiment].sort_values('confidence', ascending=False).head(2)
        print(f"\n{sentiment} message examples (highest confidence):")
        for _, row in sample.iterrows():
            print(f"- {row['text'][:50]}{'...' if len(row['text']) > 50 else ''} (Confidence: {row['confidence']:.4f})")

# Main function
def main():
    print("pengsu/MLB-care-for-mind-kor Model Analysis")
    print("------------------------------------------")
    
    print("Loading model...")
    model_name = "pengsu/MLB-care-for-mind-kor"
    # 1) 레이블 매핑 정의 (허브에서 출력된 id2label)
    label_map = {
        0: '기쁨(행복한)', 1: '고마운', 2: '설레는(기대하는)', 3: '사랑하는',
        4: '즐거운(신나는)', 5: '일상적인', 6: '생각이 많은', 7: '슬픔(우울한)',
        8: '힘듦(지침)',    9: '짜증남',      10: '걱정스러운(불안한)'
    }
    id2label = label_map
    label2id = {v: k for k, v in label_map.items()}

    # 2) 잘 알려진 베이스 모델의 config를 불러와 override
    base_model = "bert-base-multilingual-cased"
    config = AutoConfig.from_pretrained(
        base_model,
        num_labels=len(label_map),
        id2label=id2label,
        label2id=label2id
    )

    # 3) Load tokenizer for public base model (no auth token needed)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model
    )

    # 4) 허브 리포지토리에서 가중치(weights)만 덮어씌워서 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        use_auth_token=True
    )

    # pipeline 생성
    sentiment_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer
    )
    
    # Get CSV file path from user
    csv_path = input("Enter the path to your KakaoTalk CSV file: ")
    
    if not os.path.exists(csv_path):
        print("File not found. Please check the path.")
        return
    
    # Analyze the CSV file
    results_df = analyze_kakao_csv(csv_path, sentiment_pipeline)
    
    if results_df is not None:
        # Visualize the results
        print("\nVisualizing results...")
        visualize_results(results_df, "MLB_Care")
        print(f"\nAnalysis complete! Results saved to the 'results' folder.")

if __name__ == "__main__":
    main()