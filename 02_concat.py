import pandas as pd


df = pd.DataFrame() # 빈 dataframe

for i in range(1,3):
    df_temp = pd.read_csv('./crawling_data/reviews_2020_{}page.csv'.format(i))
    # 결측치 제거
    df_temp.dropna(inplace=True)
    # 중복값 제거
    df_temp.drop_duplicates(inplace=True)
    # concat
    df = pd.concat([df, df_temp], ignore_index=True)
# concat한 파일 속에 중복값이 있다면 제거
df.drop_duplicates(inplace=True)
df.info()
my_year = 2020
df.to_csv('./crawling_data/reviews_{}.csv'.format(my_year))