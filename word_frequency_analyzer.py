import pandas as pd
from collections import Counter
import re
from konlpy.tag import Okt
import os
import json
from datetime import datetime

class WordFrequencyAnalyzer:
    def __init__(self, top_n_words=10):
        self.okt = Okt()
        self.top_n_words = top_n_words
        
    def preprocess_text(self, text):
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        text = re.sub(r'\d+', '', text)
        return text.strip()
    
    def extract_words(self, text):
        # Tokenize and extract nouns
        words = self.okt.nouns(text)
        # Filter out single character words
        words = [word for word in words if len(word) > 1]
        return words
    
    def analyze_file(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.csv':
            return self._analyze_csv(file_path)
        elif file_extension == '.txt':
            return self._analyze_txt(file_path)
        else:
            raise ValueError("Unsupported file format. Please use .txt or .csv files.")
    
    def _analyze_csv(self, file_path):
        try:
            df = pd.read_csv(file_path)
            # 메시지 내용 컬럼 찾기
            content_columns = ['메시지', 'message', 'text', 'Message', 'Text', 'content', 'Content']
            found_col = None
            for col in content_columns:
                if col in df.columns:
                    found_col = col
                    break
            if found_col is None:
                # 컬럼명이 없는 경우, 마지막 컬럼을 메시지로 가정
                found_col = df.columns[-1]
            
            # 따옴표 제거 및 메시지 내용만 추출
            messages = df[found_col].astype(str).str.replace('"', '').str.replace("'", "")
            all_text = ' '.join(messages)
            return self._process_text(all_text)
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV file: {str(e)}")
        except UnicodeDecodeError:
            raise ValueError("CSV file encoding error. Please ensure the file is UTF-8 encoded")
    
    def _analyze_txt(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            # 메시지 내용만 추출
            messages = []
            for line in lines:
                # 날짜, 사용자: 메시지 형식에서 메시지만 추출
                parts = line.strip().split(',', 1)  # 첫 번째 콤마로만 분리
                if len(parts) > 1:
                    message_part = parts[1].split(':', 1)  # 첫 번째 콜론으로만 분리
                    if len(message_part) > 1:
                        messages.append(message_part[1].strip())
            
            all_text = ' '.join(messages)
            return self._process_text(all_text)
        except FileNotFoundError:
            raise FileNotFoundError(f"Text file not found: {file_path}")
        except UnicodeDecodeError:
            raise ValueError("Text file encoding error. Please ensure the file is UTF-8 encoded")
        except IOError as e:
            raise IOError(f"Error reading text file: {str(e)}")
    
    def _process_text(self, text):
        # Preprocess the text
        cleaned_text = self.preprocess_text(text)
        # Extract words
        words = self.extract_words(cleaned_text)
        # Count word frequencies
        word_freq = Counter(words)
        # Get top N most common words
        top_words = word_freq.most_common(self.top_n_words)
        return top_words

    def save_results_to_json(self, file_path, top_words):
        # results 폴더가 없으면 생성
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # JSON 형식으로 결과 구성
        result = {
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": file_path,
            "word_frequency": [
                {
                    "rank": idx + 1,
                    "word": word,
                    "count": count
                }
                for idx, (word, count) in enumerate(top_words)
            ],
            "total_unique_words": len(set(self.extract_words(self.preprocess_text(open(file_path, 'r', encoding='utf-8').read()))))
        }
        
        # 결과를 파일로 저장
        output_file = os.path.join(results_dir, f"word_frequency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return output_file 