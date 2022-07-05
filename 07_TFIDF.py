# Term Frequency-Inverse Document Frequency : 문장 내 빈도수가 높을수록 가중치 큼 / 문서 내 빈도수가 높을수록 가중치 작음

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.io import mmwrite, mmread
import pickle


# 데이터 로드
df_reviews = pd.read_csv('./reviews_2017_2022.csv')
df_reviews.info()

Tfidf = TfidfVectorizer(sublinear_tf=True)
Tfidf_matrix = Tfidf.fit_transform(df_reviews['reviews'])
print(Tfidf_matrix.shape)
# (3182, 84461) 3182개 영화,리뷰 / 84461개 단어

print(Tfidf_matrix[0])
# print(len(Tfidf_matrix[0]))


# 모델 저장
with open('./models/tfidf.pickle', 'wb') as f:
    pickle.dump(Tfidf, f)


# 매트릭스 저장
mmwrite('./models/Tfidf_movie_review.mtx', Tfidf_matrix)
