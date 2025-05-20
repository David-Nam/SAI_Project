from nlp04 import KakaoAnalyzer

def main():
    print("카카오톡 대화 분석기")
    print("-----------------")
    
    # CSV 파일 경로 입력 받기
    csv_path = input("카카오톡 CSV 파일 경로를 입력하세요: ")
    
    # 분석기 초기화 및 실행
    analyzer = KakaoAnalyzer()
    results_df = analyzer.analyze_kakao_csv(csv_path)
    
    if results_df is not None:
        print("\n결과 시각화 중...")
        analyzer.visualize_results(results_df)
        print("\n분석이 완료되었습니다! 결과는 'results' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main() 