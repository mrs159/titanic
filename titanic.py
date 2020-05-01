# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 00:14:54 2020

@author: Meng
"""

import pandas as pd #pandas处理数据
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np #numpy数学函数
from sklearn.linear_model import LinearRegression #导入线性回归
from sklearn.model_selection import KFold #导入交叉验证
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression#导入逻辑验证
from sklearn.ensemble import RandomForestClassifier#导入随机森林
from sklearn.feature_selection import SelectKBest#选择最佳特征
import re

pd.set_option('display.max_columns',None)#输出结果显示全部列
titanic = pd.read_csv("train.csv")
print(titanic.head())
print(titanic.describe())
print(titanic.info())

# 缺失值填充,Age列缺失值按中位数填充，用fillna()函数
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
print(titanic.describe())

# 把字符值转化为数值
#.loc通过自定义索引获取数据 .loc[:,:]逗号前为行，逗号后为列
titanic.loc[titanic["Sex"] == "male","Sex"] = 0
titanic.loc[titanic["Sex"] == "female","Sex"] = 1
print(titanic.describe())

#统计登船地点
print(titanic.groupby('Embarked').Name.count())
titanic["Embarked"] = titanic["Embarked"].fillna("S")

# Embarked列表处理
titanic.loc[titanic["Embarked"] == "S","Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C","Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q","Embarked"] = 2

#画散点图看分布
#sb.pairplot(titanic,hue="Age")

#选择分类特征
predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]

#将样本平均分成11份进行交叉验证
#alg = RandomForestClassifier(random_state = 10,warm_start = True,n_estimators = 26,max_depth = 6,max_features ='sqrt')
alg = RandomForestClassifier(random_state = 1,n_estimators = 18)
kf = KFold(n_splits = 11,random_state = 1)
scores = model_selection.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv=kf)
#alg = LogisticRegression(random_state = 1)
#scores = model_selection.cross_val_score(alg,titanic[predictors],titanic["survied"],cv = 11)

#print(scores)
print(scores.mean())


titanic_test = pd.read_csv("test.csv")
titanic_test['Age'] = titanic_test['Age'].fillna(titanic_test['Age'].median())
titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].median())
titanic_test.loc[titanic_test['Sex'] == 'male','Sex'] = 0
titanic_test.loc[titanic_test['Sex'] == 'female','Sex'] = 1
titanic_test['Embarked'] = titanic_test['Embarked'].fillna('S')
titanic_test.loc[titanic_test['Embarked'] == 'S', 'Embarked'] = 0
titanic_test.loc[titanic_test['Embarked'] == 'C', 'Embarked'] = 1
titanic_test.loc[titanic_test['Embarked'] == 'Q', 'Embarked'] = 2
# Initialize the algorithm class
alg = LogisticRegression(random_state=1)

# Train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])

# Make predictions using the test set.
predictions = alg.predict(titanic_test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({"PassengerId": titanic_test["PassengerId"],"Survived": predictions})
print(submission)
submission.to_csv("submission1.csv", index=False)


















