import pandas as pd

# 데이터 로드
df = pd.read_csv('./crawling_data/clean_review_2020.csv')

# 결측치 제거
df.dropna(inplace=True)
df.info()

# 영화 1개에 달린 여러 리뷰 하나로 합치기
one_sentences = []
## 영화 1개당
for title in df['title'].unique():
    temp = df[df['title'] == title]
    # 리뷰 30개로 제한
    if len(temp) > 30:
        temp = temp.iloc[:30, :]    # 30행까지
    # 최대 30개 리뷰를 하나로 결합
    one_sentence = ' '.join(temp['clean_sentences'])
    one_sentences.append(one_sentence)
df_one = pd.DataFrame({'titles':df['title'].unique(), 'reviews':one_sentences})

# 저장
df_one.to_csv('./crawling_data/clean_review_one.csv', index=False)