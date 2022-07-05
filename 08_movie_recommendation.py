import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle


# 데이터 로드
df_reviews = pd.read_csv('./reviews_2017_2022.csv')
Tfidf_matrix = mmread('./models/Tfidf_movie_review.mtx').tocsr()


# TFIDF모델 로드
with open('./models/tfidf.pickle', 'rb') as f:
    Tfidf = pickle.load(f)


# 영화 제목 / index 이용
movie_idx = df_reviews[df_reviews['titles']=='겨울왕국 2 (Frozen 2)'].index[0]
# print(movie_idx)
# print(df_reviews.iloc[1228, 1])


# 코사인 유사도
## linear_kernel : 벡터 좌표값을 이용해 3182개의 코사인값 구해줌
cosine_sim = linear_kernel(Tfidf_matrix[movie_idx], Tfidf_matrix)


# 코사인 유사도값(1에 가까울수록)으로 추천해주는 함수 생성
def getRecommendation(cosine_sim):
    simScore = list(enumerate(cosine_sim[-1]))
    ## 코사인 유사도값 기준으로 내림차순 정렬
    simScore = sorted(simScore, key=lambda  x:x[1], reverse=True)
    simScore = simScore[1:11]   # 0번은 자기자신(코사인 유사도 max)
    movieIdx = [i[0] for i in simScore]
    recMovieList = df_reviews.iloc[movieIdx, 0]
    return recMovieList

# 추천 알고리즘
recommendation = getRecommendation(cosine_sim)
print(recommendation)