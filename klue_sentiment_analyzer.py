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

class KlueSentimentAnalyzer:
    def __init__(self):
        self.model_name = "hun3359/klue-bert-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)

    def _parse_kakao_text(self, text_content):
        """카카오톡 텍스트 파일을 파싱하여 DataFrame으로 변환"""
        messages = []
        
        print("Parsing text file...")
        for line in text_content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # 메시지 라인 파싱 ({date}, {user} : {message} 형식)
            parts = line.split(',', 1)  # 첫 번째 콤마로 분리
            if len(parts) == 2:
                date = parts[0].strip()
                message_part = parts[1].strip()
                
                # 사용자와 메시지 분리
                user_message = message_part.split(':', 1)
                if len(user_message) == 2:
                    user = user_message[0].strip()
                    message = user_message[1].strip()
                    
                    messages.append({
                        'Date': date,
                        'Sender': user,
                        'Message': message
                    })
                    print(f"Found message: {date} - {user}: {message[:30]}...")
        
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
            try:
                # CSV 파일 구조: {date}, {"user"}, {"message"}
                df = pd.read_csv(file_path, encoding='utf-8')
                
                # 컬럼명이 없는 경우 기본 컬럼명 설정
                if len(df.columns) >= 3:
                    df.columns = ['Date', 'Sender', 'Message'] + list(df.columns[3:])
                
                return df
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='cp949')
                    if len(df.columns) >= 3:
                        df.columns = ['Date', 'Sender', 'Message'] + list(df.columns[3:])
                    return df
                except Exception as e:
                    print(f"Error reading CSV file: {str(e)}")
                    return None

    def analyze_sentiment(self, text):
        """텍스트의 감정을 분석"""
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

    def analyze_messages(self, file_path):
        """메시지 파일을 분석"""
        df = self._load_file(file_path)
        if df is None:
            return None

        print("File structure:", df.columns.tolist())
        
        message_col = 'Message' if 'Message' in df.columns else None
        if not message_col:
            print("Could not find message column.")
            return None
        
        timestamp_col = 'Date' if 'Date' in df.columns else None
        print(f"Using '{message_col}' as the message column.")
        if timestamp_col:
            print(f"Using '{timestamp_col}' as the timestamp column.")
        
        results = []
        print(f"Analyzing {len(df)} messages...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            message = row[message_col]
            if not message or len(message.strip()) < 2:
                continue
                
            timestamp = row[timestamp_col] if timestamp_col else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            sentiment_result = self.analyze_sentiment(message)
            if sentiment_result:
                sentiment_result['timestamp'] = timestamp
                sentiment_result['sender'] = row.get('Sender', 'Unknown')
                results.append(sentiment_result)
        
        return pd.DataFrame(results)

    def visualize_results(self, results_df):
        """분석 결과 시각화"""
        if results_df is None or len(results_df) == 0:
            print("No results to analyze.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 감정 분포 시각화
        sentiment_counts = results_df['sentiment'].value_counts()
        plt.figure(figsize=(15, 8))
        palette = sns.color_palette("husl", len(sentiment_counts))
        sentiment_counts.plot(kind='bar', color=palette)
        plt.title('감정 분석 결과')
        plt.xlabel('감정')
        plt.ylabel('메시지 수')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sentiment_analysis_{timestamp}.png')
        plt.show()
        
        # 결과 저장
        results_df.to_csv(f'{self.output_dir}/sentiment_results_{timestamp}.csv', index=False, encoding='utf-8-sig')
        
        # JSON 출력
        json_output = {
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
        
        json_path = f'{self.output_dir}/sentiment_results_{timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)
        
        # 감정별 평균 신뢰도 출력
        print("\n감정별 평균 신뢰도:")
        avg_confidence = results_df.groupby('sentiment')['confidence'].mean().reset_index()
        for _, row in avg_confidence.iterrows():
            print(f"{row['sentiment']}: {row['confidence']:.4f}")
        
        # 감정별 샘플 메시지 출력
        print("\n감정별 대표 메시지:")
        for sentiment in results_df['sentiment'].unique():
            sample = results_df[results_df['sentiment'] == sentiment].sort_values('confidence', ascending=False).head(3)
            print(f"\n{sentiment} 메시지 예시 (높은 신뢰도):")
            for _, row in sample.iterrows():
                print(f"- [{row['timestamp']}] {row['sender']}: {row['text'][:50]}{'...' if len(row['text']) > 50 else ''} (신뢰도: {row['confidence']:.4f})")
        
        print("\n결과가 저장되었습니다:")
        print(f"- CSV: {self.output_dir}/sentiment_results_{timestamp}.csv")
        print(f"- JSON: {self.output_dir}/sentiment_results_{timestamp}.json")
        print(f"- 그래프: {self.output_dir}/sentiment_analysis_{timestamp}.png") 