import os
import pandas as pd
file_list = [i for i in os.listdir("./preprocessed_data") if i.endswith(".csv") ]

import math
df_list = []
for file in file_list:
    if file != 'the outlaws.csv':
        df = pd.read_csv("./preprocessed_data/"+file, sep=",")
        df = df[["rating", "sentiment", "correlation","review","spearman"]]
        df["spearman"] = df["spearman"].abs() # 절대값으로 변환, 상관계수가 음의 방향으로 커도 상관관계의 정도가 큰 것이기 때문
        df = df.sort_values(by=['spearman'], axis=0, ascending=False)
        df = df.reset_index(drop=True)
        total_num = len(df)
        trusted_num = math.floor(total_num * 0.3)
        distrusted_num = math.floor(total_num * 0.3)
        df["label"] = -1
        for i in range(trusted_num):
            df.loc[i,"label"] = 1

        for i in df.iloc[-distrusted_num:,:].index:
            df.loc[i,"label"] = 0
        df_list.append(df[df["label"]!=-1])

result_df = pd.concat(df_list, ignore_index=True)

import codecs
import numpy as np
from tqdm import tqdm
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split

numerical_features = result_df[["rating","sentiment","correlation"]].values.reshape(-1,3,1)
text_features = result_df["review"]
text_features = np.array([np.array(t) for t in text_features])
y = result_df["label"].values.astype('int32').reshape((-1,1))

x_train, x_test,x_feature,x_test_feature, y_train, y_test = train_test_split(text_features,numerical_features, y, test_size=0.2,
                                                 random_state=34, stratify=y,shuffle=True)
x_train, x_val,x_feature,x_val_feature, y_train, y_val = train_test_split(x_train,x_feature, y_train, test_size=0.1,
                                                 random_state=134,stratify=y_train, shuffle=True)

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
bert_preprocess = hub.KerasLayer("https://tfhub.dev/jeongukjae/distilkobert_cased_preprocess/1")
bert_encoder = hub.KerasLayer("https://tfhub.dev/jeongukjae/distilkobert_cased_L-3_H-768_A-12/1")

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='review')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)
l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])

feature_input = tf.keras.layers.Input(shape=(3,1),)
feature_output = tf.keras.layers.Dense(3,activation="relu")(feature_input)
feature_output = tf.keras.layers.Flatten()(feature_output)


concatenated = tf.keras.layers.concatenate([l, feature_output])
concat_reshape = tf.keras.layers.Reshape((1,777))(concatenated) # reshape 2d to 3d
concat_out = tf.keras.layers.LSTM(32, return_sequences=True)(concat_reshape)
concat_out = tf.keras.layers.LSTM(32, return_sequences=True)(concat_out)
concat_out = tf.keras.layers.LSTM(32)(concat_out)
concat_out = tf.keras.layers.Dense(1, activation='sigmoid')(concat_out)
#concat_out = tf.keras.layers.LSTM(1, activation='sigmoid')(concatenated)

model = tf.keras.models.Model([text_input, feature_input], concat_out)



mc = ModelCheckpoint('best_review_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='adam',
 loss='binary_crossentropy',
 metrics=['acc'])
model.fit([x_train,x_feature], y_train, epochs=30,batch_size=32, callbacks=[mc],validation_data=([x_val,x_val_feature], y_val))
# multi input 이면 꼭!!! validation도 []로 묶어주는 거 잊지 말기..
def prediction_result(n):
    if n > 0.4:
        result=1
    else: result = 0
    return result

test_prediction = model.predict([x_test,x_test_feature])
y_pred = list(map(prediction_result, test_prediction))

from sklearn.metrics import f1_score, accuracy_score

#  threshikd = 0.4
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"f1 score : {f1} \n accuracy : {accuracy}")