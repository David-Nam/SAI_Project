# 우리 SAI Project
## 개요
* 여러 테스트를 하기 위한 목적으로 만든 Repository.
* 우선은 nlp04.py에 관한 내용만 업데이트 한다.
## nlp04.py 
### 입력 데이터
* 파일: csv 파일
* 파일 구조: timestamp, 이름, Message
### 출력 데이터
1. csv 파일
* sentiment(감정),confidence(정확도),text,timestamp 의 구조로 되어있음
2. json 파일
```json
{
  "model_name": "KCElectra",
  "analysis_timestamp": "20240315_143022",
  "total_messages": 123,
  "sentiment_distribution": {
    "기쁨(행복한)": 45,
    "슬픔(우울한)": 30,
    "설레는(기대하는)": 48
  },
  "average_confidence": {
    "기쁨(행복한)": 0.9234,
    "슬픔(우울한)": 0.8765,
    "설레는(기대하는)": 0.9123
  },
  "messages": [
    {
      "text": "메시지 내용",
      "sentiment": "기쁨(행복한)",
      "confidence": 0.95,
      "timestamp": "2024-03-15 14:30:22"
    }
  ]
}
```
