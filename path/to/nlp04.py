import matplotlib.pyplot as plt
import seaborn as sns

# Set font for plots (macOS 기준 한글 지원)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def visualize_results(results_df, model_name="KCElectra"):
    if results_df is None or len(results_df) == 0:
        print("No results to analyze.")
        return
    
    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Count each class and draw bar chart with dynamic colors
    sentiment_counts = results_df['sentiment'].value_counts()
    plt.figure(figsize=(10, 6))
    # 클래스 개수에 맞춰 팔레트 생성
    palette = sns.color_palette("husl", len(sentiment_counts))
    ax = sentiment_counts.plot(kind='bar', color=palette)
    plt.title(f'Sentiment Analysis Results: {model_name}')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Messages')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # 저장 및 출력
    plt.savefig(f'{output_dir}/{model_name}_sentiment_analysis.png')
    plt.show()

    # Display average confidence by sentiment
    print("\nAverage confidence by sentiment:")
    avg_confidence = results_df.groupby('sentiment')['confidence'].mean().reset_index()
    for _, row in avg_confidence.iterrows():
        print(f"{row['sentiment']}: {row['confidence']:.4f}")

    # Display sample messages with their sentiment
    print("\nSample message analysis results:")
    for sentiment in results_df['sentiment'].unique():
        sample = results_df[results_df['sentiment'] == sentiment] \
                    .sort_values('confidence', ascending=False).head(3)
        print(f"\n{sentiment} message examples (highest confidence):")
        for _, row in sample.iterrows():
            print(f"- {row['text'][:50]}{'...' if len(row['text']) > 50 else ''} (Confidence: {row['confidence']:.4f})") 