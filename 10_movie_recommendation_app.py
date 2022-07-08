import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
from scipy.io import mmread
import pickle
from PyQt5.QtCore import QStringListModel



form_window = uic.loadUiType('./movie_recommendation.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 모델 로드
        with open('./models/tfidf.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.Tfidf_matrix = mmread('./models/Tfidf_movie_review.mtx').tocsr()
        self.embedding_model = Word2Vec.load('./models/word2vec_movies_2017_2022.model')

        # 콤보박스에 영화 리스트 추가
        # self.comboBox.addItem('2017-2022 영화 리스트')
        self.df_reviews = pd.read_csv('./reviews_2017_2022.csv')
        # 모든 영화 리스트 정렬
        self.titles = list(self.df_reviews['titles'])
        self.titles.sort()
        # 콤보박스에 모든 영화 추가
        for title in self.titles:
            self.comboBox.addItem(title)
        # 콤보박스 눌렀을 때
        self.comboBox.currentIndexChanged.connect(self.combobox_slot)
        # 추천 버튼 클릭 시
        self.btn_recommendation.clicked.connect(self.btn_slot)

        # 자동완성 모델 생성
        model = QStringListModel()
        model.setStringList(self.titles)
        completer = QCompleter()
        completer.setModel(model)
        self.le_keyword.setCompleter(completer)



    def btn_slot(self):
        ## 키워드 사용
        # 사용자가 키워드 입력
        key_word = self.le_keyword.text()
        if key_word in self.titles: # 키워드 = 영화 제목
            recommendation = self.recommendation_by_movie_title(key_word)
        else:   # 키워드 ≠ 영화 제목
            recommendation = self.recommendation_by_keyword(key_word)
        # 사용자가 키워드 입력하지 않고 추천 버튼만 누른 경우를 방어
        if recommendation:
            self.lbl_recommendation.setText(recommendation)


    def getRecommendation(self, cosine_sim):
        simScore = list(enumerate(cosine_sim[-1]))
        ## 코사인 유사도값 기준으로 내림차순 정렬
        simScore = sorted(simScore, key=lambda x: x[1], reverse=True)
        simScore = simScore[:11]
        movieIdx = [i[0] for i in simScore]
        recMovieList = self.df_reviews.iloc[movieIdx, 0]
        return recMovieList


    def combobox_slot(self):
        # 사용자가 선택한 영화 제목
        title = self.comboBox.currentText()
        recommendation = self.recommendation_by_movie_title(title)
        self.lbl_recommendation.setText(recommendation)


    # 영화 제목으로 추천하는 함수
    def recommendation_by_movie_title(self, title):
        movie_idx = self.df_reviews[self.df_reviews['titles'] == title].index[0]
        # 코사인 유사도
        cosine_sim = linear_kernel(self.Tfidf_matrix[movie_idx], self.Tfidf_matrix)
        # 추천 알고리즘
        recommendation = self.getRecommendation(cosine_sim) # 리스트 형태
        recommendation = '\n'.join(list(recommendation[1:]))    # (선택한 영화 제목 빼고) 문자열(줄바꿈) 형태로 보여줌
        return recommendation


    # 키워드로 추천하는 함수
    def recommendation_by_keyword(self, keyword):
        # 사용자가 키워드 입력했을 경우
        if keyword:
            keyword = keyword.split()[0]    # 맨 앞의 키워드
            try:
                sim_word = self.embedding_model.wv.most_similar(keyword, topn=10)
            except: # 명사, 동사, 형용사/부사 아닌 키워드를 입력한 경우(ex.접속사, 감탄사, ...)
                self.lbl_recommendation.setText('명사, 동사, 형용사, 부사로 입력해주세요')
                return 0
            # 유사도값 큰 순서로 10개 단어 리스트
            words = [keyword]
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
            # 모델 적용
            sentence_vec = self.Tfidf.transform([sentence])
            # 코사인 유사도
            cosine_sim = linear_kernel(sentence_vec, self.Tfidf_matrix)
            # 추천 알고리즘
            recommendation = self.getRecommendation(cosine_sim) # 리스트 형태
            recommendation = '\n'.join(list(recommendation[:10]))    # 문자열(줄바꿈) 형태로 보여줌
            return recommendation
        # 사용자가 키워드 입력하지 않고 추천 버튼만 누른 경우
        else:
            return 0


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())
