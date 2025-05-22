from nlp04 import KakaoAnalyzer
from word_frequency_analyzer import WordFrequencyAnalyzer
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

def main():
    print("카카오톡 대화 분석기")
    print("=====================")
    
    # NLP 분석기 초기화
    print("\nNLP 모델을 사용하여 분석합니다...")
    analyzer = KakaoAnalyzer()
    
    # 파일 경로 입력 받기
    file_path = get_valid_file_path()
    
    print(f"\n'{file_path}' 파일을 분석합니다...")
    
    # 메시지 분석
    results = analyzer.analyze_kakao_csv(file_path)
    
    # 결과 시각화 및 저장
    if results is not None:
        analyzer.visualize_results(results)
        
        # 단어 빈도 분석 수행
        print("\n단어 빈도 분석을 수행합니다...")
        word_analyzer = WordFrequencyAnalyzer()
        try:
            top_words = word_analyzer.analyze_file(file_path)
            output_file = word_analyzer.save_results_to_json(file_path, top_words)
            print(f"\n단어 빈도 분석 결과가 {output_file}에 저장되었습니다.")
        except Exception as e:
            print(f"단어 빈도 분석 중 오류 발생: {str(e)}")
    else:
        print("분석을 완료할 수 없습니다.")

if __name__ == "__main__":
    main() 