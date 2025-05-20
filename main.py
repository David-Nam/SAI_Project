from klue_sentiment_analyzer import KlueSentimentAnalyzer
from nlp04 import KakaoAnalyzer
import os

def get_valid_file_path():
    """사용자로부터 유효한 파일 경로를 입력받음"""
    while True:
        file_path = input("\n분석할 파일 경로를 입력하세요 (txt 또는 csv 파일): ").strip()
        
        if not file_path:
            print("파일 경로를 입력해주세요.")
            continue
            
        if not os.path.exists(file_path):
            print("파일이 존재하지 않습니다. 다시 입력해주세요.")
            continue
            
        if not file_path.endswith(('.txt', '.csv')):
            print("지원되는 파일 형식은 .txt 또는 .csv 입니다.")
            continue
            
        return file_path

def get_model_choice():
    """사용자로부터 분석 모델 선택을 입력받음"""
    while True:
        print("\n분석 모델을 선택하세요:")
        print("1. NLP 모델 (기본 감정 분석)")
        print("2. KLUE 모델 (세부 감정 분석)")
        
        choice = input("\n선택 (1 또는 2): ").strip()
        
        if choice == "1":
            return "nlp"
        elif choice == "2":
            return "klue"
        else:
            print("잘못된 선택입니다. 1 또는 2를 입력해주세요.")

def main():
    print("카카오톡 대화 감정 분석기")
    print("=====================")
    
    # 모델 선택
    model_choice = get_model_choice()
    
    # 선택된 모델에 따라 분석기 초기화
    if model_choice == "nlp":
        print("\nNLP 모델을 사용하여 분석합니다...")
        analyzer = KakaoAnalyzer()
    else:
        print("\nKLUE 모델을 사용하여 분석합니다...")
        analyzer = KlueSentimentAnalyzer()
    
    # 파일 경로 입력 받기
    file_path = get_valid_file_path()
    
    print(f"\n'{file_path}' 파일을 분석합니다...")
    
    # 메시지 분석
    if model_choice == "nlp":
        results = analyzer.analyze_kakao_csv(file_path)
    else:
        results = analyzer.analyze_messages(file_path)
    
    # 결과 시각화 및 저장
    if results is not None:
        analyzer.visualize_results(results)
    else:
        print("분석을 완료할 수 없습니다.")

if __name__ == "__main__":
    main() 