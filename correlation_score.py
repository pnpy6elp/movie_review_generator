import os
import pandas as pd
import numpy
import json
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from pandas import json_normalize

from numpy.random import randn
from numpy.random import seed
from scipy.stats import spearmanr

cname = ["rating", "sentiment", "correlation","review","spearman","sympathy","unsympathy"]
movie_dit = {'1987':'1987',

'a taxi driver':'택시운전사',

'assassination':'암살',

'confidential assignment':'공조',

'intimate stranger':'완벽한 타인',

'the outlaws':'범죄도시'
}


for folder in tqdm(folder_list):
    data_list = []
    file_list = os.listdir(directory+"/"+folder)
    num = len(file_list)
    for i in range(0, num):
        file = directory+"/"+folder+"/%d.json" % i
        try:
            with open(file, encoding='UTF-8') as json_file:
                json_data = json.load(json_file)
                num = len(json_data["data"])
                if(num>=3): # 리뷰 수가 3개 이상인 유저들의 데이터만
                    dt = []
                    ct = 0
                    t_1 = [] # rating, predict
                    t_2 = [] # sentiment score, test
                    t_3 = []
                    for i in range(0, num):
                        if(json_data["data"][x]["title"]==movie_dit[str(folder)]):
                            if(ct == 0):
                                if(len(json_data["data"][x]["review"])<=140):
                                    dt.append(int(json_data["data"][x]["rating"])) # 현재 영화에 대한 평점
                                    dt.append(json_data["data"][x]["sentiment score"]) # 현재 영화에 대한 감성점수
                                    dt.append(abs(int(json_data["data"][x]["rating"])/10 - json_data["data"][x]["sentiment score"])) # 평점과 감성점수의 편차
                                    dt.append(json_data["data"][x]["review"]) # 현재 영화에 대한 리뷰텍스트
                                    ct += 1
                        t_1.append(int(json_data["data"][x]["rating"]))
                        t_2.append(json_data["data"][x]["sentiment score"])
                    spearman, _ = spearmanr(t_1, t_2)
                    dt.append(spearman)
                    dt.append(json_data["sympathy"])
                    dt.append(json_data["unsympathy"])
                    data_list.append(dt)

        except Exception:
            print(i, ".json", " error")
        df = pd.DataFrame(data_list, columns=cname)
        fname = "./preprocessed_data/"+folder+".csv"
        df = df.dropna(axis=0)
        df.to_csv(fname,encoding="utf-8-sig")

