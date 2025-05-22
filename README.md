# 우리 SAI Project

## 개요
* 여러 테스트를 하기 위한 목적으로 만든 Repository.
* 우선은 nlp04.py에 관한 내용만 업데이트 한다.

## nlp04.py 
### 개요
* 총 11개의 감정 분석
* 걱정스러운(불안한), 고마운 ,기쁨(행복한), 사랑하는, 생각이 많은, 설레는(기대하는), 슬픔(우울한), 일상적인, 즐거운(신나는), 짜증남,힘듦(지침)

### 입력 데이터
* 파일: csv 파일 or txt 파일
* CSV 파일 구조: Date, "User", "Message"
* TXT 파일 구조: Date, User : Message

### 출력 데이터
1. csv 파일
* sentiment(감정),confidence(정확도),text,timestamp,sender 의 구조로 되어있음
2. json 파일
```json
{
  "model_name": "KCElectra",
  "analysis_timestamp": "20240315_143022",
  "total_messages": 123,
  "sentiment_distribution": {
    "sentiment1": 45,
    "sentiment2": 30,
    "sentiment3": 48
  },
  "daily_user_analysis": {
    "2024-03-15": {
      "사용자1": {
        "sentiment_counts": {
          "일상적인": 15,
          "즐거운": 10,
          "사랑하는": 5
        },
        "sentiment_percentages": {
          "일상적인": 50.0,
          "즐거운": 33.3,
          "사랑하는": 16.7
        },
        "total_messages": 30
      },
      "사용자2": {
        "sentiment_counts": {
          "일상적인": 8,
          "즐거운": 12
        },
        "sentiment_percentages": {
          "일상적인": 40.0,
          "즐거운": 60.0
        },
        "total_messages": 20
      }
    }
  },
  "messages": [
    {
      "text": "메시지 내용",
      "sentiment": "sentiment1",
      "confidence": 0.95,
      "timestamp": "2024-03-15 14:30:22",
      "sender": "사용자명"
    }
  ]
}
```

### 추가 기능
* 감정 분석 결과 시각화 (그래프)
* 감정별 평균 신뢰도 계산
* 감정별 대표 메시지 추출
* 날짜별 사용자 감정 분석 (각 감정의 횟수와 비율)
* 결과를 CSV, JSON, PNG 형식으로 저장

## klue_sentiment_analyzer.py
### 개요
* KLUE-BERT 기반 감정 분석
* 모델: hun3359/klue-bert-base-sentiment
* 60여 가지의 감정 분석 수행

### 입력 데이터
* 파일: csv 파일 or txt 파일
* CSV 파일 구조: Date, "User", "Message"
* TXT 파일 구조: Date, User : Message

### 출력 데이터
1. csv 파일
* sentiment(감정),confidence(정확도),text,timestamp,sender 의 구조로 되어있음
2. json 파일
```json
{
  "model_name": "KLUE-BERT",
  "analysis_timestamp": "20240315_143022",
  "total_messages": 123,
  "sentiment_distribution": {
    "sentiment1": 45,
    "sentiment2": 30,
    "sentiment3": 48
  },
  "average_confidence": {
    "sentiment1": 0.9234,
    "sentiment2": 0.8765,
    "sentiment3": 0.9123
  },
  "messages": [
    {
      "text": "메시지 내용",
      "sentiment": "sentiment1",
      "confidence": 0.95,
      "timestamp": "2024-03-15 14:30:22",
      "sender": "사용자명"
    }
  ]
}
```

### 추가 기능
* 감정 분석 결과 시각화 (그래프)
* 감정별 평균 신뢰도 계산
* 감정별 대표 메시지 추출
* 결과를 CSV, JSON, PNG 형식으로 저장

## word_frequency_analyzer.py
### 개요
* 메시지 내용에서 가장 많이 언급된 단어 추출
* KoNLPy의 Okt(Open Korean Text) 형태소 분석기 사용
* 명사만 추출하여 빈도 분석
* 한 글자 단어 제외

### 입력 데이터
* 파일: csv 파일 or txt 파일
* CSV 파일 구조: Date, "User", "Message"
* TXT 파일 구조: Date, User : Message

### 출력 데이터
1. json 파일
```json
{
  "analysis_time": "2024-03-15 14:30:22",
  "file_path": "input.txt",
  "word_frequency": [
    {
      "rank": 1,
      "word": "단어1",
      "count": 100
    },
    {
      "rank": 2,
      "word": "단어2",
      "count": 80
    }
  ],
  "total_unique_words": 500
}
```

### 추가 기능
* 메시지 내용에서만 단어 추출 (날짜, 사용자 정보 제외)
* 특수문자, 숫자 제거
* 결과를 JSON 파일로 저장
* 전체 고유 단어 수 계산

