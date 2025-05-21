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

class KakaoAnalyzer:
    def __init__(self, model_name="nlp04/korean_sentiment_analysis_kcelectra"):
        self.model_name = model_name.split('/')[-1]  # 모델 이름에서 경로 부분 제거
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)

    def clean_kakao_message(self, text):
        if isinstance(text, str):
            text = re.sub(r'\[Photo\]|\[Emoticon\]|\[Video\]|\[File\]', '', text)
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            text = re.sub(r'\[Shop\]|\[Map\]', '', text)
            return text.strip()
        return ""

    def analyze_sentiment(self, text):
        if not text or len(text.strip()) < 2:
            return None
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits
            probabilities = torch.nn.functional.softmax(predictions, dim=1)
            
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            sentiment = self.model.config.id2label[predicted_class]
        
        return {"sentiment": sentiment, "confidence": confidence, "text": text}

    def _parse_kakao_text(self, text_content):
        """카카오톡 텍스트 파일을 파싱하여 DataFrame으로 변환"""
        messages = []
        current_date = None
        
        print("Parsing text file...")
        for line in text_content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # 날짜 라인 확인
            if re.match(r'\d{4}년 \d{1,2}월 \d{1,2}일', line):
                current_date = line
                print(f"Found date: {current_date}")
                continue
                
            # 메시지 라인 파싱
            match = re.match(r'(\d{4}\. \d{1,2}\. \d{1,2}\. (?:오전|오후) \d{1,2}:\d{2}), ([^:]+) : (.+)', line)
            if match:
                timestamp, sender, message = match.groups()
                messages.append({
                    'Date': current_date,
                    'Time': timestamp,
                    'Sender': sender,
                    'Message': message
                })
                print(f"Found message: {timestamp} - {sender}: {message[:30]}...")
        
        if not messages:
            print("No messages were parsed from the text file.")
            return None
            
        df = pd.DataFrame(messages)
        print(f"Successfully parsed {len(df)} messages.")
        return df

    def _load_file(self, file_path):
        """파일 형식에 따라 적절한 로딩 방식 선택"""
        if file_path.endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"Successfully read text file: {file_path}")
                return self._parse_kakao_text(content)
            except Exception as e:
                print(f"Error reading text file: {str(e)}")
                return None
        else:
            return self._load_csv(file_path)

    def _load_csv(self, csv_path):
        try:
            return pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                return pd.read_csv(csv_path, encoding='cp949')
            except (UnicodeDecodeError, pd.errors.EmptyDataError, FileNotFoundError) as e:
                print(f"Error reading CSV file: {str(e)}")
                print("Please check the CSV file encoding. Usually 'utf-8' or 'cp949'.")
                return None

    def _find_message_column(self, df):
        if df is None:
            return None
            
        possible_cols = ['Text', 'Message', 'Content', 'text', 'message', 'content']
        for col in possible_cols:
            if col in df.columns:
                return col
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].str.len().mean() > 10:
                return col
        return None

    def _find_timestamp_column(self, df):
        possible_time_cols = ['Date', 'Time', 'Timestamp', 'date', 'time', 'timestamp']
        for col in possible_time_cols:
            if col in df.columns:
                return col
        return None

    def analyze_kakao_csv(self, file_path):
        df = self._load_file(file_path)
        if df is None:
            return None

        print("File structure:", df.columns.tolist())
        
        message_col = self._find_message_column(df)
        if not message_col:
            print("Could not find a column containing message content.")
            return None
        
        timestamp_col = self._find_timestamp_column(df)
        sender_col = 'Sender' if 'Sender' in df.columns else None
        print(f"Using '{message_col}' as the message column.")
        if timestamp_col:
            print(f"Using '{timestamp_col}' as the timestamp column.")
        if sender_col:
            print(f"Using '{sender_col}' as the sender column.")
        
        df['cleaned_message'] = df[message_col].apply(self.clean_kakao_message)
        
        results = []
        print(f"Analyzing {len(df)} messages...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            message = row['cleaned_message']
            if not message or len(message.strip()) < 2:
                continue
                
            timestamp = row[timestamp_col] if timestamp_col else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sender = row[sender_col] if sender_col else "Unknown"
            
            sentiment_result = self.analyze_sentiment(message)
            if sentiment_result:
                sentiment_result['timestamp'] = timestamp
                sentiment_result['sender'] = sender
                results.append(sentiment_result)
        
        return pd.DataFrame(results)

    def visualize_results(self, results_df):
        if results_df is None or len(results_df) == 0:
            print("No results to analyze.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        sentiment_counts = results_df['sentiment'].value_counts()
        plt.figure(figsize=(10, 6))
        palette = sns.color_palette("husl", len(sentiment_counts))
        sentiment_counts.plot(kind='bar', color=palette)
        plt.title(f'Sentiment Analysis Results: {self.model_name}')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Messages')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{self.model_name}_sentiment_analysis_{timestamp}.png')
        plt.show()
        
        results_df.to_csv(f'{self.output_dir}/{self.model_name}_results_{timestamp}.csv', index=False, encoding='utf-8-sig')
        
        json_output = {
            "model_name": self.model_name,
            "analysis_timestamp": timestamp,
            "total_messages": len(results_df),
            "sentiment_distribution": sentiment_counts.to_dict(),
            "average_confidence": results_df.groupby('sentiment')['confidence'].mean().to_dict(),
            "messages": [
                {
                    "text": row['text'],
                    "sentiment": row['sentiment'],
                    "confidence": float(row['confidence']),
                    "timestamp": row['timestamp'],
                    "sender": row['sender']
                }
                for _, row in results_df.iterrows()
            ]
        }
        
        json_path = f'{self.output_dir}/{self.model_name}_results_{timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)
        
        print("\nAverage confidence by sentiment:")
        avg_confidence = results_df.groupby('sentiment')['confidence'].mean().reset_index()
        for _, row in avg_confidence.iterrows():
            print(f"{row['sentiment']}: {row['confidence']:.4f}")
        
        print("\nSample message analysis results:")
        for sentiment in results_df['sentiment'].unique():
            sample = results_df[results_df['sentiment'] == sentiment].sort_values('confidence', ascending=False).head(3)
            print(f"\n{sentiment} message examples (highest confidence):")
            for _, row in sample.iterrows():
                print(f"- [{row['timestamp']}] {row['text'][:50]}{'...' if len(row['text']) > 50 else ''} (Confidence: {row['confidence']:.4f})")
        
        print("\nResults have been saved to:")
        print(f"- CSV: {self.output_dir}/{self.model_name}_results_{timestamp}.csv")
        print(f"- JSON: {self.output_dir}/{self.model_name}_results_{timestamp}.json")
        print(f"- Plot: {self.output_dir}/{self.model_name}_sentiment_analysis_{timestamp}.png")

# Main function
def main():
    print("nlp04/kcelectra Model Analysis")
    print("-----------------------------")
    
    # Load the model and tokenizer
    print("Loading model...")
    model_name = "nlp04/korean_sentiment_analysis_kcelectra"
    analyzer = KakaoAnalyzer(model_name)
    
    # Get CSV file path from user
    csv_path = input("Enter the path to your KakaoTalk CSV file: ")
    
    if not os.path.exists(csv_path):
        print("File not found. Please check the path.")
        return
    
    # Analyze the CSV file
    results_df = analyzer.analyze_kakao_csv(csv_path)

    if results_df is not None:
        # Visualize the results
        print("\nVisualizing results...")
        analyzer.visualize_results(results_df)
        print("\nAnalysis complete! Results saved to the 'results' folder.")


if __name__ == "__main__":
    main()