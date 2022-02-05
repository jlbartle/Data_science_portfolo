#  pre-proc seinfield diolauge kaggle dataset and others
##############################################################
#  packages and data loading
##############################################################

import os
import pandas as pd
from pathlib import Path
import sklearn
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.tree import export_graphviz
from IPython.display import Image
## conda install pydotplus
import pydotplus
from sklearn.ensemble import RandomForestClassifier
from nltk.stem.porter import PorterStemmer
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import seaborn as sns

##############################################################

raw_sf = open('E:\OneDrive\DataScience\IST Text Mining\-Project-\Seinfeld Scripts.csv')
df_sf = pd.read_csv(raw_sf)
raw_sf = open('E:\OneDrive\DataScience\IST Text Mining\-Project-\SeinfeldDataRating.csv')
df_sf_rate = pd.read_csv(raw_sf)

#  drop duplicate columns.
df_sf = df_sf.drop(columns='Unnamed: 0')

##############################################################
#  clean dialogue section
##############################################################
# deletes all text within parenthesis
df_sf['Dialogue'] = df_sf['Dialogue'].str.replace(r"\(.*\)","")
df_sf['Dialogue'] = df_sf['Dialogue'].str.replace(r'\ +', ' ')
df_sf['Dialogue'] = df_sf['Dialogue'].str.replace(r'\"', ' ')
df_sf['Dialogue'] = df_sf['Dialogue'].str.replace(r'\"', ' ')
df_sf['Dialogue'] = df_sf['Dialogue'].str.replace(r'[^a-zA-Z]', " ")

# get word length of Dialogue
df_sf['Dialogue'] = df_sf['Dialogue'].astype(str)
df_sf['length'] = df_sf['Dialogue'].apply(lambda x: len(x.split(' ')))

# conver colums to str
df_sf['EpisodeNo'] = df_sf['EpisodeNo'].astype(str)
df_sf['Season'] = df_sf['Season'].astype(str)

##############################################################
# who talks the most?
##############################################################
# add up word count of all diaolauge for the 4 characters
main4 = ['GEORGE', 'JERRY', 'ELAINE', 'KRAMER']
df_sf_4 = df_sf[df_sf['Character'].isin(main4)]
df_sf_4 = df_sf_4.groupby(['Character','SEID']).sum()
total_words = df_sf_4['length'].sum()

df_sf_G = df_sf_4.T.GEORGE.transpose()
df_sf_G['C'] = np.arange(len(df_sf_G))
plt.plot('C', 'length', data=df_sf_G, label='GEORGE')

df_sf_J = df_sf_4.T.JERRY.transpose()
df_sf_J['C'] = np.arange(len(df_sf_J))
plt.plot('C', 'length', data=df_sf_J, label='JERRY')

df_sf_E = df_sf_4.T.ELAINE.transpose()
df_sf_E['C'] = np.arange(len(df_sf_E))
plt.plot('C', 'length', data=df_sf_E, label='ELAINE')

df_sf_K = df_sf_4.T.KRAMER.transpose()
df_sf_K['C'] = np.arange(len(df_sf_K))
plt.plot('C', 'length', data=df_sf_K, label='KRAMER')

plt.legend()
seasons=[['S1','S2','S3','S4','S5','S6','S7','S8','S9'],[0,5,17,40,64,86,110,134,156]]
plt.xticks(seasons[1], seasons[0], size='small')

plt.title("Words Per Episode by Character")
plt.xlabel("Episode")
plt.ylabel("Word Count")

plt.show()

plt.close()

##############################################################
# best fit plots for word count
##############################################################
seasons=[['S1','S2','S3','S4','S5','S6','S7','S8','S9'],[0,5,17,40,64,86,110,134,156]]
plt.xticks(seasons[1], seasons[0], size='small')
x,y=np.polyfit(df_sf_G['C'],df_sf_G['length'],1)
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = y + x * x_vals
plt.plot(x_vals, y_vals, '--', label='GEORGE')

x,y=np.polyfit(df_sf_J['C'],df_sf_J['length'],1)
axes = plt.gca()
y_vals = y + x * x_vals
plt.plot(x_vals, y_vals, '--', label='JERRY')

x,y=np.polyfit(df_sf_E['C'],df_sf_E['length'],1)
axes = plt.gca()
y_vals = y + x * x_vals
plt.plot(x_vals, y_vals, '--', label='ELAINE')

x,y=np.polyfit(df_sf_K['C'],df_sf_K['length'],1)
axes = plt.gca()
y_vals = y + x * x_vals
plt.plot(x_vals, y_vals, '--', label='KRAMER')
plt.legend()

plt.title("Words Per Episode by Character Regression Lines")
plt.xlabel("Episode")
plt.ylabel("Word Count")


plt.show()

plt.close()

##############################################################
# Ratings
##############################################################
## Lines
df_sf_mc = df_sf
df_sf_mc['Character'] = df_sf_mc['Character'].str.replace(r"\(.*\)","")
df_sf_mc['Character'] = df_sf_mc['Character'].str.replace(r'\ +', ' ')
df_sf_mc['Character'] = df_sf_mc['Character'].str.replace(r'\"', ' ')
df_sf_mc['Character'] = df_sf_mc['Character'].str.replace(r'\"', ' ')
df_sf_mc['Character'] = df_sf_mc['Character'].str.replace(r'[^a-zA-Z]', " ")

# get top 10 characters
df_sf['Character'].value_counts()[:11].index.tolist()
mainc = df_sf['Character'].value_counts()[:11].index.tolist()
del mainc[9]
## https://www.cinemablend.com/television/20-Best-Seinfeld-Characters-Ranked-Order-69982.html
df_sf_mc = df_sf_mc[df_sf_mc['Character'].isin(mainc)]

# gather number of lines per character per episode
n_lines_df = df_sf_mc.groupby(['Season','EpisodeNo','Character'])\
    .count()['Dialogue']\
    .reset_index()\
    .sort_values(['Season','EpisodeNo','Dialogue'])\
    .reset_index(drop=True)
n_lines_df.columns = ['Season','Episode','Character','n_lines']

##############################################################
# Ratings
##############################################################
# verify episodes are sorted correctly
assert all(df_sf_rate.sort_values(by="SEID") == df_sf_rate)

# attach episode numbers
n_seasons = df_sf_rate['Season'].max()
n_episodes = df_sf_rate.shape[0]
episode_num = [1]*n_episodes
last_ep = 0
season_of_last_ep = 0
for i in range(n_episodes):
    season_of_this_ep = df_sf_rate['Season'][i]
    if season_of_this_ep > season_of_last_ep:
        this_ep = 1
    else:
        this_ep = last_ep + 1
    last_ep = this_ep
    season_of_last_ep = season_of_this_ep
    episode_num[i] = this_ep
df_sf_rate['EpisodeNo'] = episode_num
rating_per_episode_df = df_sf_rate[['Season','EpisodeNo','Rating']]
# rename columns
rating_per_episode_df.columns = ['Season','Episode','Rating']

## Join dataframes

# set correct types
n_lines_df['Episode'] = n_lines_df['Episode'].astype(int)
n_lines_df['Season'] = n_lines_df['Season'].astype(int)

ratings_and_lines_df = pd.merge(n_lines_df,
                                rating_per_episode_df,
                                on=['Season','Episode'],
                                how='right')

ratings_and_lines_df['Season_Episode'] = list(zip(ratings_and_lines_df['Season'].values,
         ratings_and_lines_df['Episode'].values))


# get correlations and p values
combined_df = ratings_and_lines_df.copy()
coefs = []
p_vals = []
for character in mainc:
    char_df = combined_df.loc[combined_df['Character']==character,:]
    y = char_df["Rating"]
    X = char_df["n_lines"]
    y = y[~np.isnan(X)]
    X = X[~np.isnan(X)]
    coefs.append(pearsonr(X,y)[0])
    p_vals.append(pearsonr(X,y)[1])
coef_dict = {'Character': mainc, 'Coefficient': coefs, 'p Value': p_vals}
coef_df = pd.DataFrame.from_dict(coef_dict).sort_values('Coefficient', ascending=False).reset_index(drop=True)

# place '*' next to character name if p < .1
coef_df['Character*'] = [coef_df['Character'][i] + '*' if coef_df['p Value'][i] > .1 else coef_df['Character'][i] + '' for i in range(len(coef_df['p Value']))]
coef_df.to_html('temp.html')
# https://stackoverflow.com/questions/36271302/changing-color-scale-in-seaborn-bar-plot
def colors_from_values(values, palette_name):
    indices = pd.Series(range(len(values)))
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

fig, ax = plt.subplots()
fig.set_size_inches(64, 32)
sns.despine()
ax.set_ylim(-.54,.49)
sns.barplot(x=coef_df["Character*"], y=coef_df["Coefficient"], ax=ax, orient='v', palette=colors_from_values(coef_df["Coefficient"], "RdYlGn"))
plt.xlabel('\nCharacter', size=52)
plt.ylabel('Pearson Correlation Coefficient', size=52)
plt.xticks(rotation=0, size=28)
plt.yticks(size=44)
plt.title('Character Lines Correlation to IMBD Rating', size=3, loc='left', pad=300)
fig.savefig("influence_lines.png", bbox_inches='tight')

combined_df = ratings_and_lines_df.copy()
for character in coef_df['Character*']:
    char_df = combined_df.loc[combined_df['Character'] == ''.join([x for x in character if x != '*']), :]
    y = char_df["Rating"]
    X = char_df["n_lines"]
    y = y[~np.isnan(X)]
    X = X[~np.isnan(X)]
    m, b = np.polyfit(X, y, 1)

    plt.plot(X, y, '.')
    plt.plot(X, m * X + b)
    plt.xlabel('Number of Lines in Episode')
    plt.ylabel('Episode IMDb Rating')
    plt.title(f'{character}')
    plt.show()
    plt.savefig(str(f'{character}').replace('*','')+'.png')
    plt.close()




##############################################################
# repeat correlation calculation using word count
##############################################################


# gather number of words per character per episode
n_words_df = df_sf_mc.groupby(['Season','EpisodeNo','Character'])\
    .sum()['length']\
    .reset_index()\
    .sort_values(['Season','EpisodeNo'])\
    .reset_index(drop=True)
n_words_df.columns = ['Season','Episode','Character','n_words']

##############################################################
# Ratings
##############################################################
# verify episodes are sorted correctly
assert all(df_sf_rate.sort_values(by="SEID") == df_sf_rate)

# attach episode numbers
n_seasons = df_sf_rate['Season'].max()
n_episodes = df_sf_rate.shape[0]
episode_num = [1]*n_episodes
last_ep = 0
season_of_last_ep = 0
for i in range(n_episodes):
    season_of_this_ep = df_sf_rate['Season'][i]
    if season_of_this_ep > season_of_last_ep:
        this_ep = 1
    else:
        this_ep = last_ep + 1
    last_ep = this_ep
    season_of_last_ep = season_of_this_ep
    episode_num[i] = this_ep
df_sf_rate['EpisodeNo'] = episode_num
rating_per_episode_df = df_sf_rate[['Season','EpisodeNo','Rating']]
# rename columns
rating_per_episode_df.columns = ['Season','Episode','Rating']

## Join dataframes

# set correct types
n_words_df['Episode'] = n_words_df['Episode'].astype(int)
n_words_df['Season'] = n_words_df['Season'].astype(int)

ratings_and_lines_df = pd.merge(n_words_df,
                                rating_per_episode_df,
                                on=['Season','Episode'],
                                how='right')

ratings_and_lines_df['Season_Episode'] = list(zip(ratings_and_lines_df['Season'].values,
         ratings_and_lines_df['Episode'].values))


# get correlations and p values
combined_df = ratings_and_lines_df.copy()
coefs = []
p_vals = []
for character in mainc:
    char_df = combined_df.loc[combined_df['Character']==character,:]
    y = char_df["Rating"]
    X = char_df["n_words"]
    y = y[~np.isnan(X)]
    X = X[~np.isnan(X)]
    coefs.append(pearsonr(X,y)[0])
    p_vals.append(pearsonr(X,y)[1])
coef_dict = {'Character': mainc, 'Coefficient': coefs, 'p Value': p_vals}
coef_df = pd.DataFrame.from_dict(coef_dict).sort_values('Coefficient', ascending=False).reset_index(drop=True)

# place '*' next to character name if p < .1
coef_df['Character*'] = [coef_df['Character'][i] + '*' if coef_df['p Value'][i] > .1 else coef_df['Character'][i] + '' for i in range(len(coef_df['p Value']))]
coef_df.to_html('tempWC.html')
# https://stackoverflow.com/questions/36271302/changing-color-scale-in-seaborn-bar-plot
def colors_from_values(values, palette_name):
    indices = pd.Series(range(len(values)))
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

fig, ax = plt.subplots()
fig.set_size_inches(64, 32)
sns.despine()
ax.set_ylim(-.54,.49)
sns.barplot(x=coef_df["Character*"], y=coef_df["Coefficient"], ax=ax, orient='v', palette=colors_from_values(coef_df["Coefficient"], "RdYlGn"))
plt.xlabel('\nCharacter', size=52)
plt.ylabel('Pearson Correlation Coefficient', size=52)
plt.xticks(rotation=0, size=28)
plt.yticks(size=44)
plt.title('Character Lines Correlation to IMBD Rating', size=3, loc='left', pad=300)
fig.savefig("influence_words.png", bbox_inches='tight')

combined_df = ratings_and_lines_df.copy()
for character in coef_df['Character*']:
    char_df = combined_df.loc[combined_df['Character'] == ''.join([x for x in character if x != '*']), :]
    y = char_df["Rating"]
    X = char_df["n_words"]
    y = y[~np.isnan(X)]
    X = X[~np.isnan(X)]
    m, b = np.polyfit(X, y, 1)

    plt.plot(X, y, '.')
    plt.plot(X, m * X + b)
    plt.xlabel('Number of Words in Episode')
    plt.ylabel('Episode IMDb Rating')
    plt.title(f'{character}')
    plt.savefig(str(f'{character}').replace('*','')+'wc.png')
    plt.show()
    plt.close()


##############################################################
# models?
##############################################################
# get rid of all lines of dialoge with less than 10 words.
# for vectorizing get rid of low word count lines
cond = df_sf['length'] > 10
df_sf = df_sf[cond]

#  join dfs
master_sf_df = pd.merge(
    df_sf,
    df_sf_rate,
    how="inner",
    on='SEID',
    left_on=None,
    right_on=None,
    left_index=True,
    right_index=False,
    sort=True,
    suffixes=("_x", "_y"),
    copy=False,
    indicator=False,
    validate=None,
)
master_sf_df = master_sf_df.reset_index()
master_sf_df = master_sf_df.drop(columns='index')
master_sf_df = master_sf_df.drop(columns='Description')

## Tokenize and Vectorize the Dialogue
## Create the list of dia
DiaL = []
for next in master_sf_df['Dialogue']:
    DiaL.append(next)

print(DiaL)


#####################################################################
### Vectorize
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
MyCountV = CountVectorizer(input="content", lowercase=True, stop_words="english")

MyDTM = MyCountV.fit_transform(DiaL)  # create a sparse matrix
print(type(MyDTM))
# vocab is a vocabulary list
vocab = MyCountV.get_feature_names()  # change to a list

MyDTM = MyDTM.toarray()  # convert to a regular array
print(list(vocab)[10:20])
ColumnNames = MyCountV.get_feature_names()
MyDTM_DF = pd.DataFrame(MyDTM, columns=ColumnNames)
print(MyDTM_DF)


# We will be creating new data frames - one for NB and one for Bern.
## These are the two new and currently empty DFs
FinalDF_STEM = pd.DataFrame()
FinalDF_TFIDF = pd.DataFrame()

MyVect = CountVectorizer(input='content',
                              analyzer='word',
                              stop_words='english',
                              ##stop_words=["and", "or", "but"],
                              # token_pattern='(?u)[a-zA-Z]+',
                              # token_pattern=pattern,
                              # strip_accents = 'unicode',
                              lowercase=True
                              )

from tqdm import tqdm

for name in ['Larry David', 'Jerry Seinfeld']:
    builder = name + "DF"
    builderB = name + "DFB"

    X1 = MyVect.fit_transform(master_sf_df[master_sf_df['Writers'].isin([name])])
    X2 = MyVect.fit_transform(master_sf_df[master_sf_df['Writers'].isin([name])])

    ColumnNames1 = MyVect.get_feature_names()
    NumFeatures1 = len(ColumnNames1)
    ColumnNames2 = MyVect.get_feature_names()
    NumFeatures2 = len(ColumnNames2)

    builderS = pd.DataFrame(X1.toarray(), columns=ColumnNames1)
    builderT = pd.DataFrame(X2.toarray(), columns=ColumnNames2)

    builderS["Label"] = name
    builderT["Label"] = name

    FinalDF_STEM = FinalDF_STEM.append(builderS)
    FinalDF_TFIDF = FinalDF_TFIDF.append(builderT)

## Replace the NaN with 0 because it actually
## means none in this case
FinalDF_STEM = FinalDF_STEM.fillna(0)
FinalDF_TFIDF = FinalDF_TFIDF.fillna(0)
##########################################################
def RemoveNums(SomeDF):
    # print(SomeDF)
    print("Running Remove Numbers function....\n")
    temp = SomeDF
    MyList = []
    for col in temp.columns:
        Logical2 = str.isalpha(col)  ## this checks for anything
        if (Logical2 == False):  # or Logical2==True):
            MyList.append(str(col))
    temp.drop(MyList, axis=1, inplace=True)
    return temp
## Call the function ....
FinalDF_STEM = RemoveNums(FinalDF_STEM)
FinalDF_TFIDF = RemoveNums(FinalDF_TFIDF)

print(FinalDF_STEM)  # 1
print(FinalDF_TFIDF)  # 2

# gen train test data
TrainDF1, TestDF1 = train_test_split(FinalDF_STEM, test_size=10)
print(TrainDF1)
print(TestDF1)
TrainDF2, TestDF2 = train_test_split(FinalDF_TFIDF, test_size=10)

###############################################
## For all three DFs - separate LABELS
#################################################
## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
### TEST ---------------------
Test1Labels = TestDF1["Label"]
print(Test1Labels)
Test2Labels = TestDF2["Label"]

print(Test2Labels)
## remove labels
TestDF1 = TestDF1.drop(["Label"], axis=1)
TestDF2 = TestDF2.drop(["Label"], axis=1)

print(TestDF1)

## TRAIN ----------------------------
Train1Labels = TrainDF1["Label"]
Train2Labels = TrainDF2["Label"]


## remove labels
TrainDF1 = TrainDF1.drop(["Label"], axis=1)
TrainDF2 = TrainDF2.drop(["Label"], axis=1)


## confusion matrix
from sklearn.metrics import confusion_matrix

#############################################
###########  SVM ############################
#############################################
# from sklearn.svm import LinearSVC
SVM_Model = LinearSVC(C=1)
SVM_Model.fit(TrainDF1, Train1Labels)

# print("SVM prediction:\n", SVM_Model.predict(TestDF1))
# print("Actual:")
# print(Test1Labels)

SVM_matrix = confusion_matrix(Test1Labels, SVM_Model.predict(TestDF1))
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")


TRAIN = TrainDF1  ## As noted above - this can also be TrainDF2, etc.
TRAIN_Labels = Train1Labels
TEST = TestDF1
TEST_Labels = Test1Labels

SVM_Model1 = LinearSVC(C=1)
SVM_Model1.fit(TRAIN, TRAIN_Labels)

# print("SVM prediction:\n", SVM_Model1.predict(TEST))
# print("Actual:")
# print(TEST_Labels)

SVM_matrix = confusion_matrix(TEST_Labels, SVM_Model1.predict(TEST))
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")
# --------------other kernels
## RBF------------------------------------------
##------------------------------------------------------
SVM_Model2 = sklearn.svm.SVC(C=100, kernel='rbf',
                             verbose=True, gamma="auto")
SVM_Model2.fit(TRAIN, TRAIN_Labels)

# print("SVM prediction:\n", SVM_Model2.predict(TEST))
# print("Actual:")
# print(TEST_Labels)

SVM_matrix2 = confusion_matrix(TEST_Labels, SVM_Model2.predict(TEST))
print("\nThe confusion matrix is:")
print(SVM_matrix2)
print("\n\n")

##-----------------------------------------
## POLY
##_--------------------------------------------------
SVM_Model3 = sklearn.svm.SVC(C=100, kernel='poly', degree=2,
                             gamma="auto", verbose=True)

print(SVM_Model3)
SVM_Model3.fit(TRAIN, TRAIN_Labels)

# print("SVM prediction:\n", SVM_Model3.predict(TEST))
# print("Actual:")
# print(TEST_Labels)

SVM_matrix3 = confusion_matrix(TEST_Labels, SVM_Model3.predict(TEST))
print("\nThe confusion matrix is:")
print(SVM_matrix3)
print("\n\n")

###########  broken #########################
#function used for showing most influental features
#def show_most_informative_features(vectorizer, clf, n=20):
#    feature_names = vectorizer.get_feature_names()
#    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
#    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
#    for (coef_1, fn_1), (coef_2, fn_2) in top:
#        print("\t%.4f\t%-15s" % (coef_2, fn_2))

#show_most_informative_features(MyVect,SVM_Model1)
#show_most_informative_features(MyVect,SVM_Model2)
#show_most_informative_features(MyVect,SVM_Model3)
