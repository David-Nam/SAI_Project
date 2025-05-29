#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sentiment_analysis.py

한국어 감정 분석 스크립트
  - 입력: .csv 또는 .txt 파일 (--csv, --txt)
  - 출력: 감정분류 결과 CSV, JSON (--output_csv, --output_json)
  - 사용 모델: monologg/koelectra-small-v3-discriminator (기본)
"""

import re
import json
import argparse
from datetime import datetime
import pandas as pd
import torch
from transformers import ElectraForSequenceClassification, ElectraTokenizer
from tqdm import tqdm

# 6가지 감정 레이블 (모델 학습 시 사용한 순서와 동일해야 합니다)
LABELS = ["기쁨", "슬픔", "분노", "중립", "당황", "혐오"]

def read_csv_file(path):
    """
    .csv 형식: 
      date, "user", "message"
    예) 2025-05-07 17:56:56,"남준우","어떻할까?"
    """
    df = pd.read_csv(
        path,
        names=["timestamp", "user", "message"],
        header=None,
        parse_dates=["timestamp"],
        encoding="utf-8-sig"
    )
    return df

def parse_kakao_datetime(dt_str):
    """
    카카오톡 .txt timestamp 파싱
      예) "2022. 7. 18. 오후 3:32"
      → datetime 객체 반환, 파싱 실패 시 None
    """
    s = dt_str.strip()
    # 년, 월, 일 제거
    s = s.replace("년", ".").replace("월", ".").replace("일", "").strip()
    # 오전/오후 처리
    ampm = None
    if "오전" in s or "오후" in s:
        if "오전" in s:
            ampm = "AM"
            s = s.replace("오전", "")
        else:
            ampm = "PM"
            s = s.replace("오후", "")
    try:
        # 예: "2022. 7. 18.  3:32" → "%Y. %m. %d. %H:%M"
        dt = datetime.strptime(s, "%Y. %m. %d. %H:%M")
        if ampm == "PM" and dt.hour < 12:
            dt = dt.replace(hour=dt.hour + 12)
        if ampm == "AM" and dt.hour == 12:
            dt = dt.replace(hour=0)
        return dt
    except ValueError:
        # ISO 형식 등 다른 포맷 시도
        try:
            return datetime.fromisoformat(dt_str)
        except:
            return None

def read_txt_file(path):
    """
    .txt 형식:
      date, user : message
    예) 2022. 7. 18. 오후 3:32, 이신영 : 웅웅...별로긴해
    """
    records = []
    pattern = re.compile(r'^(.+?),\s*([^:]+)\s*:\s*(.*)$')
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            m = pattern.match(line)
            if not m:
                continue
            dt_str, user, msg = m.groups()
            dt = parse_kakao_datetime(dt_str)
            if dt is None:
                continue
            records.append({
                "timestamp": dt,
                "user": user.strip(),
                "message": msg.strip()
            })
    return pd.DataFrame(records)

def preprocess_inputs(df):
    """
    - timestamp, message가 없는 행 제거
    - message 내부 whitespace 정리
    """
    df = df.dropna(subset=["timestamp", "message"])
    df["text"] = df["message"].str.replace(r'\s+', ' ', regex=True).str.strip()
    return df

def classify(df, model_name, batch_size=32, device="cpu"):
    """
    df: timestamp, user, message, text 컬럼 존재
    반환: sentiment, confidence, text, timestamp, sender 컬럼의 DataFrame
    """
    # 토크나이저/모델 로딩
    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    model = ElectraForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS)
    ).to(device)
    model.eval()

    results = {"sentiment": [], "confidence": [], "text": [], "timestamp": [], "sender": []}
    texts = df["text"].tolist()
    times = df["timestamp"].tolist()
    users = df["user"].tolist()

    for i in tqdm(range(0, len(texts), batch_size), desc="감정 분류"):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        for idx, text in enumerate(batch_texts):
            results["sentiment"].append(LABELS[pred[idx].item()])
            results["confidence"].append(float(conf[idx].item()))
            results["text"].append(text)
            results["timestamp"].append(times[i+idx].strftime("%Y-%m-%d %H:%M:%S"))
            results["sender"].append(users[i+idx])

    return pd.DataFrame(results)

def generate_json(df_out, model_name):
    """
    - 전체 메시지 수
    - 감정 분포
    - 일별/사용자별 통계
    - 메시지 리스트
    """
    total = len(df_out)
    analysis_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dist = df_out["sentiment"].value_counts().to_dict()

    # 일별 사용자별 통계
    df_out["date"] = pd.to_datetime(df_out["timestamp"]).dt.date.astype(str)
    daily = {}
    for date, grp in df_out.groupby("date"):
        daily[date] = {}
        for user, sub in grp.groupby("sender"):
            cnt = sub["sentiment"].value_counts().to_dict()
            perc = {k: round(v/len(sub)*100, 1) for k, v in cnt.items()}
            daily[date][user] = {
                "sentiment_counts": cnt,
                "sentiment_percentages": perc,
                "total_messages": len(sub)
            }

    messages = df_out[["text","sentiment","confidence","timestamp","sender"]].to_dict(orient="records")

    return {
        "model_name": model_name,
        "analysis_timestamp": analysis_ts,
        "total_messages": total,
        "sentiment_distribution": dist,
        "daily_user_analysis": daily,
        "messages": messages
    }

def main():
    parser = argparse.ArgumentParser(description="한국어 감정 분석 스크립트")
    parser.add_argument("--csv",       help="입력 .csv 파일 경로")
    parser.add_argument("--txt",       help="입력 .txt 파일 경로")
    parser.add_argument("--model",     help="HuggingFace 모델명", 
                        default="monologg/koelectra-small-v3-discriminator")
    parser.add_argument("--output_csv", help="출력 CSV 파일명", default="output.csv")
    parser.add_argument("--output_json",help="출력 JSON 파일명", default="output.json")
    args = parser.parse_args()

    # 입력 파일 읽기
    dfs = []
    if args.csv:
        dfs.append(read_csv_file(args.csv))
    if args.txt:
        dfs.append(read_txt_file(args.txt))
    if not dfs:
        print("오류: --csv 또는 --txt 중 하나 이상 지정해야 합니다.")
        return

    # 통합 및 전처리
    df = pd.concat(dfs, ignore_index=True)
    df = preprocess_inputs(df)

    # 분류 실행
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df_out = classify(df, args.model, device=device)

    # CSV 저장
    df_out.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ CSV 저장: {args.output_csv}")

    # JSON 생성 및 저장
    result_json = generate_json(df_out, args.model)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON 저장: {args.output_json}")

if __name__ == "__main__":
    main()