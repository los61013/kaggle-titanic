import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from IPython.display import display
from IPython.display import display_html
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
#survied:是否生存(0死1活) pclass:艙等(1頭等2商務3經濟) name:姓名
#sex:性別(0女1男) age:年齡 sibsp:船上有兄弟/配偶的數目 parch:船上有父母/小孩的數目
#ticket:船票編號 fare:船票價格 cabin:船艙編號 embarked:上船的岸口(C法國瑟堡Q英國皇后鎮S英國南安普頓)

#資料路徑
test_data = pd.read_csv(r"C:\Users\Faker\Desktop\資料探勘\DEMO\titanic data\test.csv")
train_data = pd.read_csv(r"C:\Users\Faker\Desktop\資料探勘\DEMO\titanic data\train.csv")
save_path = r"C:\Users\Faker\Desktop\資料探勘\DEMO\titanic data"
save_name = r"alldata.csv"
print(train_data.shape)
print(test_data.shape)
#資料合併後儲存
alldata = train_data.append(test_data)
alldata.reset_index(inplace=True, drop=True)
#alldata.to_csv(save_path+'\\'+ save_name,encoding="utf_8_sig",index=False)
print(alldata.head())

#計算缺失
all_data_miss = alldata.isnull().sum()
print(all_data_miss[1:11])
total_data = np.product(alldata.shape)
total_miss = all_data_miss[1:11].sum()
percent_miss = (total_miss / total_data) * 100
print(total_miss, percent_miss)

#性別存活率
plt.figure(num=1)
plt.title("sex/survived")
#sn.histplot(x= alldata['sex'], hue=alldata['survived'])
sn.countplot(x= alldata['sex'], hue=alldata['survived'])
display(alldata[["sex", "survived"]].groupby(['sex'], as_index=False).mean().round)

#艙等存活率
plt.figure(num=2)
plt.title("pclass/survived")
sn.countplot(x=alldata['pclass'], hue=alldata['survived'])
display(alldata[['pclass', 'survived']].groupby(['pclass'], as_index=False).mean().round)

#性別與艙等存活率
display(alldata[['pclass', 'sex', 'survived']].groupby(['pclass', 'sex'], as_index=False).mean().sort_values(by='survived', ascending=False))

##艙等/票價 存活率
plt.figure(num=3)
plt.title("fare & pclass vs survived")
#alldata['logfare'] = (alldata['fare']+1).map(lambda x : np.log10(x) if x > 0 else 0) #取log
#sn.boxplot(x=alldata['fare'],y=alldata['pclass'],hue=alldata['survived'],orient='h',palette="Set3")
sn.scatterplot(x=alldata['pclass'], y=alldata['fare'], hue=alldata['survived'])
display(pd.pivot_table(alldata, values=['fare'], index=['pclass'], columns=['survived'], aggfunc='median').round(3))


#上船口存活率
plt.figure(num=4)
plt.title("embarked/survived")
sn.countplot(x=alldata['embarked'], hue=alldata['survived'])
display(alldata[['embarked', 'survived']].groupby(['embarked'], as_index=False).mean().round)

#上船口艙等
plt.figure(num=5)
plt.title("embarked/pclass")
#sn.histplot(x=alldata['embarked'], hue=alldata['pclass'])
sn.countplot(x=alldata['embarked'], hue=alldata['pclass'])
display(alldata[['embarked', 'pclass']].groupby(['embarked'], as_index=False).mean().round)
#display(alldata[['pclass', 'embarked', 'sex']].groupby(['pclass', 'embarked'], as_index=False).mean().sort_values(by='sex', ascending=False))

#姓名長度存活率
namelen = []
for names in alldata['name']:
    namelen.append(len(names.split()))
alldata['namelen'] = namelen
plt.figure(num=6)
plt.title("namelen/survived")
sn.countplot(x=alldata['namelen'], hue=alldata['survived'])
display(alldata[['namelen', 'survived']].groupby(['namelen'], as_index=False).mean().round)

#家庭人數存活率
plt.figure(num=7)
plt.title("family/survived")
alldata['family'] = alldata['sibsp']+alldata['parch']
print(alldata['family'])
sn.countplot(x=alldata['family'], hue=alldata['survived'])
display(alldata[['family', 'survived']].groupby(['family'], as_index=False).mean().round)

#年齡特徵
plt.figure(num=8)
sn.scatterplot(x=alldata['age'], y=alldata['fare'], hue=alldata['survived'])
plt.figure(num=9)
sn.histplot(x=alldata['age'], hue=alldata['survived'],bins=10)
alldata['haveage'] = alldata['age'].isnull().map(lambda x: 0 if x == True else 1)
fig, [ax1,ax2] = plt.subplots(1, 2)
ax1 = sn.countplot(x=alldata['pclass'], hue=alldata['haveage'],ax=ax1)
ax2 = sn.countplot(x=alldata['sex'], hue=alldata['haveage'],ax=ax2)
pd.crosstab(alldata['haveage'], alldata['sex'], margins=True).round(3)

haveage_s = ((alldata['haveage'] == 1) & (alldata['pclass'] != 3) & (alldata['survived'] == 1))
haveage_d = ((alldata['haveage'] == 1) & (alldata['pclass'] != 3) & (alldata['survived'] == 0))
fig, ax = plt.subplots()
plt.title("age/survived in pclass1.2")
ax = sn.distplot(alldata.loc[haveage_s, 'age'], kde=False, bins=10, label='survived')
ax = sn.distplot(alldata.loc[haveage_d, 'age'], kde=False, bins=10, label='dead')
ax.legend()

#資料清洗
#年齡
alldata['title'] = alldata['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
alldata['title'] = alldata['title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Dona', 'Jonkheer',
                                                'Major', 'Rev', 'Sir'], 'Rare')
alldata['title'] = alldata['title'].replace(['Mlle', 'Ms', 'Mme'], 'Miss')
alldata['title'] = alldata['title'].replace(['Lady'], 'Mrs')
alldata['title'] = alldata['title'].map({"Mr":0, "Rare":1, "Master":2, "Miss":3, "Mrs":4})
meamage = alldata.groupby('title')['age'].median()
#alldata['newage'] = alldata['age']
for i in range(0,5):
    alldata.loc[(alldata['age'].isnull()) & (alldata['title'] == i), 'age'] = meamage[i]
#alldata['newage'] = alldata['newage'].astype('int')

alldata['name'] = alldata['namelen']
alldata["sex"] = [1 if i == "male" else 0 for i in alldata["sex"]]
alldata["fare"] = alldata["fare"].fillna(alldata["fare"].mean())
alldata['fare'] = (alldata['fare']+1).map(lambda x : np.log10(x) if x > 0 else 0) #取log
alldata["embarked"] = [0 if i == "S" else i for i in alldata["embarked"]]
alldata["embarked"] = [1 if i == "Q" else i for i in alldata["embarked"]]
alldata["embarked"] = [2 if i == "C" else i for i in alldata["embarked"]]
alldata["embarked"] = alldata["embarked"].fillna(0)
alldata = alldata.drop('cabin',axis=1)
alldata = alldata.drop('ticket',axis=1)
alldata = alldata.drop('haveage',axis=1)
alldata = alldata.drop('parch',axis=1)
alldata = alldata.drop('sibsp',axis=1)
alldata = alldata.drop('namelen',axis=1)
print(alldata.head())
all_data_miss = alldata.isnull().sum()
print(all_data_miss[1:15])
#alldata.to_csv(save_path+'\\'+ save_name,encoding="utf_8_sig",index=False)

#資料圖關聯熱點圖
cols = ['survived', 'pclass', 'name', 'sex', 'age', 'family', 'fare', 'embarked','title']
alldata_corr = alldata[cols].corr()
print(alldata_corr)
plt.figure(num=12)
plt.title("att/corr")
sn.heatmap(alldata_corr,annot=True)
#plt.show()

#ML
#獨立資料
Y_train = train_data['survived']
X_train = alldata.drop("survived", axis=1)
X_test = X_train.iloc[891:]
X_train = X_train.head(891)

#隨機森林
random_forest = RandomForestClassifier()#oob_score=True)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train), 3)
#oobscore = round(random_forest.oob_score_, 3)
#print("oobscore:", oobscore)
print(acc_random_forest)

#邏輯回歸
logreg = LogisticRegression(solver='lbfgs',max_iter=200)
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train), 2)

#SVM
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train), 2)

#決策數
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train), 2)

#perceptron
perceptron = Perceptron(max_iter=100)
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train), 2)

#SGD
sgd = linear_model.SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
sgd.score(X_train, Y_train)
acc_sgd = round(sgd.score(X_train, Y_train), 2)

#貝式分類器
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train), 2)

#結果排序
results = pd.DataFrame({
    "model": ["Support Vector Machines", "Logistic Regression", "Random Forest", "Naive Bayes", "Perceptron",
              "Stochastic Gradient Decent", "Decision Tree"],
    "acc": [acc_linear_svc, acc_log, acc_random_forest, acc_gaussian, acc_perceptron, acc_sgd,
            acc_decision_tree]})
result = results.sort_values(by="acc", ascending=False)
result = result.set_index("acc")
print(result.head(9))

#交叉驗證
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
acc = cross_val_score(rf, X_train, Y_train, cv=9, scoring = "accuracy")
print("Acc:", acc)
print("Mean:", acc.mean())
print("Standard Deviation:", acc.std())

#特徵觀察
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print(importances.head(9))

#超參數調整
random_forest = RandomForestClassifier(criterion='gini',
                                        n_estimators=100,
                                        min_samples_split=8,
                                        min_samples_leaf=2,
                                        oob_score=True,
                                        random_state=1,
                                        )
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
oobscore = round(random_forest.oob_score_,3)
print("oobscore:", oobscore)






#https://yulongtsai.medium.com/https-medium-com-yulongtsai-titanic-top3-8e64741cc11f
#http://www.taroballz.com/2019/05/25/ML_RandomForest_Classifier/
#https://chih-sheng-huang821.medium.com/%E4%BA%A4%E5%8F%89%E9%A9%97%E8%AD%89-cross-validation-cv-3b2c714b18db
#https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8