import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle



# 1. 영화 제목 / index 이용한 영화 추천 알고리즘
# 데이터 로드
df_reviews = pd.read_csv('./token_reviews_2017_2022.csv')

Tfidf_matrix = mmread('./models/Tfidf_movie_review.mtx(2)').tocsr()


# TFIDF모델 로드
with open('./models/tfidf(2).pickle', 'rb') as f:
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
print(recommendation[1:11])




# 2. 키워드 이용한 추천 알고리즘
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle
from gensim.models import Word2Vec


# 모델 로드
embedding_model = Word2Vec.load('./models/word2vec_movies_2017_2022.model')


# 키워드 지정
keyword = '마블'
sim_word = embedding_model.wv.most_similar(keyword, topn=10)


# 유사도값 큰 순서로 10개 단어 리스트
words = []
for word, _ in sim_word:
    words.append(word)

# sentence = [words[0]] * 10 + [word[1] * 9]
##           [유사도 max 키워드] * 10 + ...
sentence = []
count = 10
for word in words:
    sentence = sentence + [word] * count
    count -= 1
sentence = ' '.join(sentence)
# print(sentence)
# 커밍 커밍 커밍 커밍 커밍 커밍 커밍 커밍 커밍 커밍 마블 마블 마블 마블 마블 마블 마블 마블 마블 아이언맨 아이언맨 아이언맨 아이언맨 아이언맨 아이언맨 아이언맨 아이언맨 베놈 베놈 베놈 베놈 베놈 베놈 베놈 프롬 프롬 프롬 프롬 프롬 프롬 엔드게임 엔드게임 엔드게임 엔드게임 엔드게임 파커 파커 파커 파커 어벤져스 어벤져스 어벤져스 스파이디 스파이디 멀티버스


# 모델 적용
sentence_vec = Tfidf.transform([sentence])


# 코사인 유사도
cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)


# 코사인 유사도값(1에 가까울수록)으로 추천해주는 함수 생성
def getRecommendation(cosine_sim):
    simScore = list(enumerate(cosine_sim[-1]))
    ## 코사인 유사도값 기준으로 내림차순 정렬
    simScore = sorted(simScore, key=lambda  x:x[1], reverse=True)
    simScore = simScore[:11]   # 0번(코사인 유사도 max, 키워드 자기자신)은 처음부터 제외됐었으니 다시 포함
    movieIdx = [i[0] for i in simScore]
    recMovieList = df_reviews.iloc[movieIdx, 0]
    return recMovieList


# 추천 알고리즘
recommendation = getRecommendation(cosine_sim)
print(recommendation[:10])




# 3. 문장 이용한 추천 알고리즘
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle
from konlpy.tag import Okt
import re


# 문장 이용
sentence = '화려한 액션과 소름돋는 반전이 있는 영화'


## tokenizer 로드
okt = Okt()

## 영화와 직접적으로 관련된 stopword 추가
df_stopwords = pd.read_csv('./crawling_data/stopwords.csv')
stopwords = list(df_stopwords['stopword'])
stopwords = stopwords + ['영화', '연출', '관객', '관람', '출연', '개봉', '개봉일', '주인공', '출연', '배우', '리뷰', '촬영', '각본', '극장', '감독', '네이버', '박스오피스', '박스', '오피스', '장면', '내용', '연기', '되어다']


## 한글 제외 모두 제거
review = re.sub('[^가-힣 ]', ' ', sentence)  # review 문장 속 한글, 띄어쓰기 제외 모두 띄어쓰기로 대체


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
        words.append(word)
    # 형태소 리스트 + 품사 리스트
clean_sentence = ' '.join(words)
print(clean_sentence)


# 모델 적용
sentence_vec = Tfidf.transform([sentence])


# 코사인 유사도
cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)


# 코사인 유사도값(1에 가까울수록)으로 추천해주는 함수 생성
def getRecommendation(cosine_sim):
    simScore = list(enumerate(cosine_sim[-1]))
    ## 코사인 유사도값 기준으로 내림차순 정렬
    simScore = sorted(simScore, key=lambda  x:x[1], reverse=True)
    simScore = simScore[:11]   # 0번(코사인 유사도 max, 키워드 자기자신)은 처음부터 제외됐었으니 다시 포함
    movieIdx = [i[0] for i in simScore]
    recMovieList = df_reviews.iloc[movieIdx, 0]
    return recMovieList


# 추천 알고리즘
recommendation = getRecommendation(cosine_sim)
print(recommendation[:10])


