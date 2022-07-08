import pandas as pd
from gensim.models import Word2Vec


# 데이터 로드
review_word = pd.read_csv('./token_reviews_2017_2022.csv')
review_word.info()

# 리뷰 리스트 만들기
clean_token_reviews = list(review_word['reviews'])
print(clean_token_reviews[0])

# 리뷰 리스트 속 형태소만 추출한 리스트 만들기
clean_tokens = []
for sentence in clean_token_reviews:
    token = sentence.split()    # 리뷰를 띄어쓰기 기준으로 split
    clean_tokens.append(token)
print(clean_tokens[0])

# 벡터라이징
embedding_model = Word2Vec(clean_tokens,
                           vector_size=100,   # 100차원으로 축소
                           window=4,    # 4개씩 끊어서 학습 ==> Conv1D의 커널사이즈 지정
                           min_count=20,    # 빈도 최소 20번 이상인 단어들만
                           workers=4,   # cpu 코어 개수
                           epochs=100,  # 100번 학습
                           sg=1)    # 어떤 알고리즘 사용하는지 (skip gram algorithm)

# 모델 저장
embedding_model.save('./models/word2vec_movies_2017_2022(2).model')

# {형태소 : 차원 좌표}
print(list(embedding_model.wv.index_to_key))    # 형태소 출력
print(len(embedding_model.wv.index_to_key)) # 형태소 개수(=차원 축소 전 원래 차원의 개수(22539)) 출력
