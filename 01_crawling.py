from selenium import webdriver
import pandas as pd
from selenium.common.exceptions import NoSuchElementException
import time


options = webdriver.ChromeOptions()
driver = webdriver.Chrome('./chromedriver.exe', options=options)
options.add_argument('lang=ko_KR')

# 2020년 영화 디렉토리 url 주소
# https://movie.naver.com/movie/sdb/browsing/bmovie.naver?open=2020&page=1 ~ 37페이지까지

# 제목 Xpath (1페이지당 20개)
# //*[@id="old_content"]/ul/li[1]/a
# //*[@id="old_content"]/ul/li[2]/a
# //*[@id="old_content"]/ul/li[3]/a
# //*[@id="old_content"]/ul/li[20]/a

# 리뷰 카테고리 Xpath
review_button_xpath = '//*[@id="movieEndTabMenu"]/li[6]/a'
review_xpath_number = '//*[@id="reviewTab"]/div/div/div[2]/span/em' # 리뷰 총 몇개인지

# 리뷰 페이지 Xpath
# //*[@id="pagerTagAnchor1"]
# //*[@id="pagerTagAnchor2"]
# //*[@id="pagerTagAnchor10"]

# 리뷰 제목 Xpath
review_title_xpath = '//*[@id="reviewTab"]/div/div/ul/li[1]/a'
                     # '# //*[@id="reviewTab"]/div/div/ul/li[10]/a'

# 리뷰 Xpath
# //*[@id="content"]/div[1]/div[4]/div[1]
review_xpath = '//*[@id="content"]/div[1]/div[4]/div[1]/div[4]'

your_year = 2020    # 할당받은 연도로 수정
# for i in range(1,38):
for i in range(13,38):
    url = 'https://movie.naver.com/movie/sdb/browsing/bmovie.naver?open=2020&page={}'.format(i)
    titles = []
    reviews = []
    try:
        # driver.get(url)
        ## 제목
        for j in range(1,21):
            driver.get(url)
            time.sleep(0.5)
            movie_title_xpath = '//*[@id="old_content"]/ul/li[{}]/a'.format(j)
            try:    # 영화 1개 에러
                title = driver.find_element("xpath", movie_title_xpath).text    # xpath의 텍스트만 받아오기
                driver.find_element("xpath", movie_title_xpath).click() # 영화 제목 클릭
                time.sleep(0.5)
                ## 리뷰
                driver.find_element("xpath", review_button_xpath).click()   # 리뷰 버튼 클릭
                time.sleep(0.5)
                review_range = driver.find_element("xpath", review_xpath_number).text
                review_range = review_range.replace(',', '')    # 리뷰가 총 1,000이상일 때 숫자에서 ,제거
                review_range = (int(review_range)-1) // 10 + 2  # 리뷰가 1페이지에 10개씩 존재
                # 리뷰 모든 페이지
                for k in range(1, review_range):
                    # 리뷰 ?페이지 버튼 누르기
                    review_page_button_xpath = '//*[@id="pagerTagAnchor{}"]'.format(k)
                    try:
                        # 리뷰 버튼 xpath가 다른 경우
                        try:
                            driver.find_element("xpath", review_page_button_xpath).click()  # 리뷰 페이지 버튼 클릭
                        except:
                            driver.find_element("xpath", '//*[@id="movieEndTabMenu"]/li[5]/a').click()  # 리뷰 페이지 버튼 클릭
                        # 리뷰 한 페이지마다
                        for l in range(1, 11):
                            back_flag = False
                            review_title_xpath = '//*[@id="reviewTab"]/div/div/ul/li[{}]/a'.format(l)
                            try:
                                review = driver.find_element("xpath", review_title_xpath).click()   # 리뷰 제목 클릭
                                back_flag = True
                                time.sleep(0.5)
                                review = driver.find_element("xpath", review_xpath).text
                                # print(title)
                                # print(review)
                                titles.append(title)
                                reviews.append(review)
                                driver.back()   # 뒤로가기
                            except:
                                if back_flag:
                                    driver.back()
                                print("review : ", i, j, k, l)
                        driver.back()   # 뒤로가기
                    except:
                        print("review page : ", i, j, k)

            except:
                print('movie : ', i, j)
        df = pd.DataFrame({'title':titles, 'reviews':reviews})
        df.to_csv('./crawling_data/MovieReview_2020/reviews_{}_{}page.csv'.format(your_year, i), index=False)
    except:
        print('page : ', i) #몇번째 페이지에서 에러났는지
driver.close()
