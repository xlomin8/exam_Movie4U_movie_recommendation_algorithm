import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from gensim.models import Word2Vec
from sklearn.manifold import TSNE


# 폰트 지정
font_path = './malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()

## x,y축 -값 정렬
mpl.rcParams['axes.unicode_minus'] = False
## 한글 출력
rc('font', family=font_name)


# 모델 로드
embedding_model = Word2Vec.load('./models/word2vec_movies_2017_2022(2).model')


# 지정한 키워드와 가장 유사한 값 10개
key_word = '마블'
sim_word = embedding_model.wv.most_similar(key_word, topn=10)   # [(형태소, 유사도)] # most_silimar : 벡터 공간에서 비슷한 위치에 있는 애들 모아서 보여줌
print(sim_word)


# 유사도 높은 값 찾기
vectors = []
labels = []
for label, _ in sim_word:
    labels.append(label)
    vectors.append(embedding_model.wv[label])   # 100차원 좌표값

## 100차원 -> 차원 축소
df_vectors = pd.DataFrame(vectors)

tsne_model = TSNE(perplexity=40,
                  n_components=2,   # 2차원으로 축소
                  init='pca',   # 차원 축소 알고리즘
                  n_iter=2500)  # 2500번 학습 => epoch

# 2차원
new_value = tsne_model.fit_transform(df_vectors)
df_xy = pd.DataFrame({'words':labels,
                      'x':new_value[:, 0],  # x좌표
                      'y':new_value[:, 1]}) # y좌표

print(df_xy)
print(df_xy.shape)


# 시각화
## 키워드 중심
# df_xy.loc[10] = (key_word, 0, 0)
df_xy.loc[df_xy.shape[0]] = (key_word, 0, 0)

## 키워드 = 별
plt.figure(figsize=(8,8))
plt.scatter(0,0, s=1500, marker='*')
## 키워드에서 뻗어나가는 선 그리기
for i in range(len(df_xy)-1):
    a = df_xy.loc[[i, 10]]  # 10은 키워드
    plt.plot(a.x, a.y, '-D', linewidth=1)
    # 유사도 높은 단어 10개 표시
    plt.annotate(df_xy.words[i],
                 xytext=(1,1),
                 xy=(df_xy.x[i], df_xy.y[i]),
                 textcoords='offset points',    # 좌표에서 살짝 띄우기
                 ha = 'right',  # 오른쪽 정렬
                 va = 'bottom') # 수직 정렬

plt.show()