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
import platform

# Set font for plots (한글 지원)
system = platform.system()
if system == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif system == 'Linux':
    plt.rcParams['font.family'] = 'NanumGothic'
else:  # Windows or others
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class KakaoAnalyzer:
    def __init__(self, model_name="nlp04/korean_sentiment_analysis_kcelectra"):
        self.model_name = model_name.split('/')[-1]  # 모델 이름에서 경로 부분 제거
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)  # 모델을 GPU로 이동
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
        """
        주어진 텍스트의 감정을 분석하는 함수
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            dict: 감정 분석 결과를 담은 딕셔너리
                - sentiment: 감정 레이블
                - confidence: 예측 신뢰도
                - text: 원본 텍스트
            None: 텍스트가 너무 짧거나 비어있는 경우
        """
        if not text or len(text.strip()) < 2:
            return None
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # 입력을 GPU로 이동
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits
            probabilities = torch.nn.functional.softmax(predictions, dim=1)
            
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            sentiment = self.model.config.id2label[predicted_class]
        
        return {"sentiment": sentiment, "confidence": confidence, "text": text}

    def _parse_kakao_text(self, text_content):
        """
        카카오톡 텍스트 파일을 파싱하여 DataFrame으로 변환하는 함수
        
        Args:
            text_content (str): 카카오톡 대화 내용이 담긴 텍스트
            
        Returns:
            pd.DataFrame: 파싱된 메시지를 담은 데이터프레임
                - Date: 메시지 날짜
                - Sender: 발신자
                - Message: 메시지 내용
            None: 파싱된 메시지가 없는 경우
            
        Note:
            - 입력 텍스트는 '{date}, {user} : {message}' 형식이어야 함
            - 각 라인은 개행문자로 구분됨
            - 빈 라인은 무시됨
        """
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
            return self._load_csv(file_path)

    def _load_csv(self, csv_path):
        try:
            # CSV 파일 구조: {date}, {"user"}, {"message"}
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # 컬럼명이 없는 경우 기본 컬럼명 설정
            if len(df.columns) >= 3:
                df.columns = ['Date', 'Sender', 'Message'] + list(df.columns[3:])
            
            return df
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_path, encoding='cp949')
                if len(df.columns) >= 3:
                    df.columns = ['Date', 'Sender', 'Message'] + list(df.columns[3:])
                return df
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
        """
        카카오톡 대화 파일을 분석하여 감정 분석 결과를 반환하는 함수
        
        Args:
            file_path (str): 분석할 카카오톡 대화 파일의 경로
            
        Returns:
            pd.DataFrame: 감정 분석 결과를 담은 데이터프레임
            None: 파일 로드 실패 시
        """
        # 파일 로드
        df = self._load_file(file_path)
        if df is None:
            return None

        # 파일 구조 출력
        print("File structure:", df.columns.tolist())
        
        # 메시지 컬럼 찾기
        message_col = self._find_message_column(df)
        if not message_col:
            print("Could not find a column containing message content.")
            return None
        
        # 타임스탬프와 발신자 컬럼 찾기
        timestamp_col = self._find_timestamp_column(df)
        sender_col = 'Sender' if 'Sender' in df.columns else None
        
        # 사용할 컬럼 정보 출력
        print(f"Using '{message_col}' as the message column.")
        if timestamp_col:
            print(f"Using '{timestamp_col}' as the timestamp column.")
        if sender_col:
            print(f"Using '{sender_col}' as the sender column.")
        
        # 메시지 전처리
        df['cleaned_message'] = df[message_col].apply(self.clean_kakao_message)
        
        # 감정 분석 결과를 저장할 리스트
        results = []
        print(f"Analyzing {len(df)} messages...")
        
        # 각 메시지에 대해 감정 분석 수행
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            message = row['cleaned_message']
            # 빈 메시지나 너무 짧은 메시지는 건너뛰기
            if not message or len(message.strip()) < 2:
                continue
                
            # 타임스탬프와 발신자 정보 추출
            timestamp = row[timestamp_col] if timestamp_col else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sender = row[sender_col] if sender_col else "Unknown"
            
            # 감정 분석 수행
            sentiment_result = self.analyze_sentiment(message)
            if sentiment_result:
                # 결과에 타임스탬프와 발신자 정보 추가
                sentiment_result['timestamp'] = timestamp
                sentiment_result['sender'] = sender
                results.append(sentiment_result)
        
        # 결과를 데이터프레임으로 변환하여 반환
        return pd.DataFrame(results)

    def _create_visualization(self, sentiment_counts, timestamp):
        plt.figure(figsize=(10, 6))
        palette = sns.color_palette("husl", len(sentiment_counts))
        sentiment_counts.plot(kind='bar', color=palette)
        plt.title(f'Sentiment Analysis Results: {self.model_name}')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Messages')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{self.model_name}_sentiment_analysis_{timestamp}.png')
        plt.close()

    def _convert_korean_date(self, date_str):
        try:
            if '오후' in date_str:
                date_str = date_str.replace('오후', '')
                hour = int(date_str.split(':')[0].split()[-1])
                if hour < 12:
                    hour += 12
                date_str = date_str.replace(str(hour), f"{hour:02d}")
            elif '오전' in date_str:
                date_str = date_str.replace('오전', '')
            
            date_obj = datetime.strptime(date_str.strip(), '%Y. %m. %d. %H:%M')
            return date_obj.date()
        except Exception as e:
            print(f"Error parsing date: {date_str}, Error: {str(e)}")
            return None

    def _analyze_daily_user_sentiment(self, results_df):
        """
        일별 사용자별 감정 분석 결과를 계산하는 함수
        
        Args:
            results_df (pd.DataFrame): 감정 분석 결과가 담긴 데이터프레임
                - timestamp: 메시지 시간
                - sender: 발신자
                - sentiment: 감정 레이블
                
        Returns:
            dict: 일별 사용자별 감정 분석 결과
                {
                    'YYYY-MM-DD': {
                        'user_name': {
                            'sentiment_counts': {감정: 개수},
                            'sentiment_percentages': {감정: 비율},
                            'total_messages': 총 메시지 수
                        }
                    }
                }
        """
        daily_user_analysis = {}
        results_df['date'] = results_df['timestamp'].apply(self._convert_korean_date)
        
        for date, date_group in results_df.groupby('date'):
            if date is None:
                continue
                
            date_str = date.strftime('%Y-%m-%d')
            daily_user_analysis[date_str] = {}
            
            for user, user_group in date_group.groupby('sender'):
                sentiment_counts = user_group['sentiment'].value_counts()
                total_messages = len(user_group)
                sentiment_percentages = (sentiment_counts / total_messages * 100).round(1)
                
                daily_user_analysis[date_str][user] = {
                    "sentiment_counts": sentiment_counts.to_dict(),
                    "sentiment_percentages": sentiment_percentages.to_dict(),
                    "total_messages": total_messages
                }
        return daily_user_analysis

    def _print_analysis_summary(self, results_df, sentiment_counts):
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

    def visualize_results(self, results_df):
        if results_df is None or len(results_df) == 0:
            print("No results to analyze.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sentiment_counts = results_df['sentiment'].value_counts()
        
        # Create and save visualization
        self._create_visualization(sentiment_counts, timestamp)
        
        # Save CSV results
        results_df.to_csv(f'{self.output_dir}/{self.model_name}_results_{timestamp}.csv', index=False, encoding='utf-8-sig')
        
        # Analyze daily user sentiment
        daily_user_analysis = self._analyze_daily_user_sentiment(results_df)
        
        # Prepare and save JSON output
        json_output = {
            "model_name": self.model_name,
            "analysis_timestamp": timestamp,
            "total_messages": len(results_df),
            "sentiment_distribution": sentiment_counts.to_dict(),
            "daily_user_analysis": daily_user_analysis,
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
        
        # Print analysis summary
        self._print_analysis_summary(results_df, sentiment_counts)
        
        print("\nResults have been saved to:")
        print(f"- CSV: {self.output_dir}/{self.model_name}_results_{timestamp}.csv")
        print(f"- JSON: {self.output_dir}/{self.model_name}_results_{timestamp}.json")
        print(f"- Plot: {self.output_dir}/{self.model_name}_sentiment_analysis_{timestamp}.png")