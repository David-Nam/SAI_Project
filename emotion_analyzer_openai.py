import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import platform
from typing import List, Dict, Union
import openai
from tqdm import tqdm
import logging
import re
from tenacity import retry, stop_after_attempt, wait_exponential

# Set font for plots (한글 지원)
system = platform.system()
if system == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif system == 'Linux':
    plt.rcParams['font.family'] = 'NanumGothic'
else:  # Windows or others
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionAnalyzerOpenAI:
    def __init__(self, api_key: str):
        """
        OpenAI API를 사용한 감정 분석기 초기화
        
        Args:
            api_key (str): OpenAI API 키
        """
        self.api_key = api_key
        openai.api_key = api_key
        
        # 감정 레이블 정의
        self.emotion_labels = [
            "일상적인",
            "즐거운(신나는)",
            "기쁨(행복한)",
            "슬픔(우울한)",
            "짜증남",
            "설레는(기대하는)",
            "생각이 많은",
            "걱정스러운(불안한)",
            "사랑하는",
            "고마운"
        ]
        
        # 출력 디렉토리 생성
        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)

    def parse_kakao_datetime(self, dt_str: str) -> datetime:
        """카카오톡 날짜/시간 파싱"""
        s = dt_str.strip()
        s = s.replace("년", ".").replace("월", ".").replace("일", "").strip()
        
        ampm = None
        if "오전" in s or "오후" in s:
            if "오전" in s:
                ampm = "AM"
                s = s.replace("오전", "")
            else:
                ampm = "PM"
                s = s.replace("오후", "")
        
        try:
            dt = datetime.strptime(s, "%Y. %m. %d. %H:%M")
            if ampm == "PM" and dt.hour < 12:
                dt = dt.replace(hour=dt.hour + 12)
            if ampm == "AM" and dt.hour == 12:
                dt = dt.replace(hour=0)
            return dt
        except ValueError:
            try:
                return datetime.fromisoformat(dt_str)
            except:
                return None

    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if not isinstance(text, str):
            return ""
        
        # 기본 전처리
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[Photo\]|\[Emoticon\]|\[Video\]|\[File\]', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\[Shop\]|\[Map\]', '', text)
        
        return text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def analyze_emotion_openai(self, text: str) -> Dict[str, float]:
        """
        OpenAI API를 사용하여 텍스트의 감정 분석
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            Dict[str, float]: 감정별 점수
        """
        if not text or len(text.strip()) < 2:
            return None

        prompt = f"""다음 메시지의 감정을 분석해주세요. 각 감정에 대해 0.0에서 1.0 사이의 점수를 부여해주세요.
감정 목록: {', '.join(self.emotion_labels)}

메시지: {text}

JSON 형식으로 응답해주세요. 예시:
{{
    "일상적인": 0.1,
    "즐거운(신나는)": 0.8,
    ...
}}"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a Korean emotion analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # JSON 파싱
            result = json.loads(response.choices[0].message.content)
            
            # 점수 정규화
            total = sum(result.values())
            if total > 0:
                result = {k: v/total for k, v in result.items()}
            
            return result
            
        except Exception as e:
            logger.error(f"Error in emotion analysis: {str(e)}")
            return None

    def analyze_emotions_batch(self, texts: List[str], batch_size: int = 10) -> List[Dict[str, float]]:
        """배치 단위로 감정 분석 수행"""
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="감정 분석 중"):
            batch_texts = texts[i:i+batch_size]
            batch_results = []
            
            for text in batch_texts:
                processed_text = self.preprocess_text(text)
                emotion_scores = self.analyze_emotion_openai(processed_text)
                batch_results.append(emotion_scores)
                time.sleep(0.5)  # API 호출 제한 고려
            
            results.extend(batch_results)
        
        return results

    def analyze_kakao_file(self, file_path: str) -> pd.DataFrame:
        """카카오톡 대화 파일 분석"""
        # 파일 읽기
        if file_path.endswith('.txt'):
            df = self._parse_kakao_text(file_path)
        else:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            if len(df.columns) >= 3:
                df.columns = ['timestamp', 'sender', 'message'] + list(df.columns[3:])
        
        if df is None or len(df) == 0:
            return None
        
        # 전처리
        df = df.dropna(subset=['timestamp', 'message'])
        df['text'] = df['message'].apply(self.preprocess_text)
        
        # 배치 처리로 감정 분석
        texts = df['text'].tolist()
        emotion_scores_list = self.analyze_emotions_batch(texts)
        
        # 결과 데이터프레임 생성
        results = []
        for idx, row in df.iterrows():
            if idx < len(emotion_scores_list):
                emotion_scores = emotion_scores_list[idx]
                if emotion_scores:
                    dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
                    results.append({
                        'text': row['text'],
                        'emotion_scores': emotion_scores,
                        'dominant_emotion': dominant_emotion,
                        'timestamp': row['timestamp'],
                        'sender': row['sender']
                    })
        
        return pd.DataFrame(results)

    def _parse_kakao_text(self, file_path: str) -> pd.DataFrame:
        """카카오톡 텍스트 파일 파싱"""
        records = []
        pattern = re.compile(r'^(.+?),\s*([^:]+)\s*:\s*(.*)$')
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    m = pattern.match(line)
                    if not m:
                        continue
                    
                    dt_str, user, msg = m.groups()
                    dt = self.parse_kakao_datetime(dt_str)
                    if dt is None:
                        continue
                    
                    records.append({
                        "timestamp": dt,
                        "sender": user.strip(),
                        "message": msg.strip()
                    })
            
            return pd.DataFrame(records)
            
        except Exception as e:
            logger.error(f"Error parsing text file: {str(e)}")
            return None

    def visualize_results(self, results_df: pd.DataFrame):
        """분석 결과 시각화 및 저장"""
        if results_df is None or len(results_df) == 0:
            print("No results to analyze.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 감정 분포 시각화
        emotion_counts = results_df['dominant_emotion'].value_counts()
        plt.figure(figsize=(12, 6))
        palette = sns.color_palette("husl", len(emotion_counts))
        emotion_counts.plot(kind='bar', color=palette)
        plt.title('Emotion Analysis Results')
        plt.xlabel('Emotion')
        plt.ylabel('Number of Messages')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/emotion_analysis_{timestamp}.png')
        plt.close()
        
        # CSV 저장
        results_df.to_csv(
            f'{self.output_dir}/emotion_analysis_{timestamp}.csv',
            index=False,
            encoding='utf-8-sig'
        )
        
        # JSON 저장
        json_output = {
            "analysis_timestamp": timestamp,
            "total_messages": len(results_df),
            "emotion_distribution": emotion_counts.to_dict(),
            "messages": results_df.to_dict(orient='records')
        }
        
        with open(f'{self.output_dir}/emotion_analysis_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)
        
        print("\nResults have been saved to:")
        print(f"- CSV: {self.output_dir}/emotion_analysis_{timestamp}.csv")
        print(f"- JSON: {self.output_dir}/emotion_analysis_{timestamp}.json")
        print(f"- Plot: {self.output_dir}/emotion_analysis_{timestamp}.png")

def main():
    print("OpenAI Emotion Analysis")
    print("----------------------")
    
    # API 키 입력
    api_key = input("Enter your OpenAI API key: ")
    
    # 분석기 초기화
    analyzer = EmotionAnalyzerOpenAI(api_key)
    
    # 파일 경로 입력
    file_path = input("Enter the path to your KakaoTalk file (txt or csv): ")
    
    if not os.path.exists(file_path):
        print("File not found. Please check the path.")
        return
    
    # 파일 분석
    results_df = analyzer.analyze_kakao_file(file_path)
    
    if results_df is not None:
        # 결과 시각화
        print("\nVisualizing results...")
        analyzer.visualize_results(results_df)
        print("\nAnalysis complete! Results saved to the 'results' folder.")

if __name__ == "__main__":
    main() 