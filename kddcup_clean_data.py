#!/usr/bin/python

from __future__ import division

import os

import pandas as pd

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

df = pd.read_csv('corrected.csv')

# 将label归一化成01
# df['type_status'][df['type_status'] != 'normal.'] = 1
# df['type_status'][df['type_status'] == 'normal.'] = 0


class_col = df.pop('type_status')

# 离散类型归一化
cat_sel = [n for n in df.columns]  # 类别特征数值化
for column in cat_sel:
    df[column] = pd.factorize(df[column].values, sort=True)[0]


# 标准化
# df_norm = (df - df.min()) / (df.max() - df.min()) * 10
df_norm = df.fillna(0)
# temp=df_norm.corr()


df_norm.insert(0, 'type_status', class_col)

# 随机抽取1k条数据
df_norm = df_norm.sample(n=1000, frac=None, replace=False, weights=None, random_state=None, axis=0)
df_norm.to_csv("data/kddcup_1k.csv", index=False, header=False)
