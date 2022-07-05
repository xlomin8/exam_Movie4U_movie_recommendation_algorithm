import pandas as pd
import glob



df = pd.DataFrame() # 빈 dataframe
data_paths = glob.glob('./clean_review_one/*')
print(data_paths)

for path in data_paths:
    df_temp = pd.read_csv(path)
    # 결측치 제거
    df_temp.dropna(inplace=True)
    # 중복값 제거
    df_temp.drop_duplicates(inplace=True)
    # concat
    df = pd.concat([df, df_temp], ignore_index=True)
# concat한 파일 속에 중복값이 있다면 제거
df.drop_duplicates(inplace=True)
df.info()

# my_year = 2020
df.to_csv('./reviews_2017_2022.csv', index=False)