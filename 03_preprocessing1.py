import pandas as pd
from konlpy.tag import Okt
import re

# 데이터 로드
df = pd.read_csv('./crawling_data/reviews_2019.csv')
df.info()

# 데이터 전처리
## tokenizer 로드
okt = Okt()

## stopword 로드
df_stopwords = pd.read_csv('./crawling_data/stopwords.csv')
stopwords = list(df_stopwords['stopword'])

## 한글 제외 모두 제거
count = 0
clean_sentences = []
for review in df.reviews:
    count += 1
    if count % 10 == 0:
        print('.', end='')
    if count % 100 == 0:
        print()
    review = re.sub('[^가-힣 ]', ' ', review)  # review 문장 속 한글, 띄어쓰기 제외 모두 띄어쓰기로 대체
    ## 특정 단어로 도배된(20개 이상) 리뷰 제거(2019)
    # review = review.split()
    # words = []
    # for word in review:
    #     if len(word) > 20:
    #         word = ' '
    #         words.append(word)
    # review = ' '.join(words)

    ## 형태소 분리
    token = okt.pos(review, stem=True)  # pos -> (형태소, 품사)형태로 분리
    df_token = pd.DataFrame(token, columns=['word', 'class'])
    df_token = df_token[(df_token['class'] == 'Noun') |
                        (df_token['class'] == 'Verb') |
                        (df_token['class'] == 'Adjective')] # 명사, 동사, 형용사/부사만 남아 있는 dataframe 만들기
    ## 불용어 제거
    words = []
    for word in df_token.word:
        ## 한 글자 제거
        if len(word) > 1:
            if word not in stopwords:
                words.append(word)
    # 형태소 리스트 + 품사 리스트
    clean_sentence = ' '.join(words)
    clean_sentences.append(clean_sentence)  # [형태소 + 품사]
# clean_sentences 컬럼 추가
df['clean_sentences'] = clean_sentences
# review 컬럼 제거
df = df[['title', 'clean_sentences']]

# 결측치 제거
df.dropna(inplace=True)

# 저장
df.to_csv('./crawling_data/clean_review_2019.csv', index=False)


