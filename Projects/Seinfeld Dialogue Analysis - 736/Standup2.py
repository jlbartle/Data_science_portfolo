#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 18:55:37 2021

Author:    Juan Ventosa
Course:    IST 736 Text Mining
Prof:      Dr. Ami Gates
Grp Proj:  Seinfeld Standup
Date:      3/18/2021

"""
#%%############# LOAD PACKAGES ###################

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
import random as rd
import numpy as np
import re
import json
import requests 
import os
import string


#%%##############  LOAD DATA ##################

# Load ScriptDF.csv from previously saved data
ScriptDF = pd.read_csv(
    '/Users/Juan/Desktop/Python/Spyder Projects/IST 736 Projects/Group Project/data/ScriptDF.csv')


#%% Use WordCount to identify long dialogue lines

TempDF = ScriptDF.copy()
LongLines = TempDF.loc[TempDF['WordCount'] > 50]

# Check value counts for LongLines
print(LongLines['Label'].value_counts())

# Extract Jerry's lines (Stand-up?)
Standup = LongLines.loc[LongLines['Label'] == 'JERRY']


#%% Clean and stem Standup Dialogue

# Create Empty lists
StandupDialogue = []


for line in Standup['Dialogue']:
    StandupDialogue.append(line)


#%% Use StandupDialogue to create a stemlist and cleanlist

PS = PorterStemmer()

#Create Stemming function


def Stemmer(Dialogue, stemlist, cleanlist):

    for row in Dialogue:
        # iteratively save each row as a list of tokens separated by spaces
        Tokens = row.split(" ")

        Wordlist1 = []  # empty list to temporarily store clean Tokens
        Wordlist2 = []  # empty list to temporarilty store stem Tokens

        for item in Tokens:  # additional cleaning
            item = item.lstrip()  # remove spaces to the left
            item = re.sub('\d', "", item)  # remove digits
            item = re.sub('\n', "", item)  # remove new line
            item = re.sub("\(", "", item)  # remove open parenthesis
            item = re.sub("\'", "", item)  # remove single quotes
            item = re.sub("[...]", " ", item)  # replace ... with space
            item = re.sub("\.", "", item)  # remove periods
            item = re.sub("\,", "", item)  # remove commas
            item = re.sub("\)", "", item)  # remove close parenthesis
            Wordlist1.append(item)
            item = PS.stem(item)  # stem each item using PorterStemmer
            Wordlist2.append(item)  # append cleaned tokens to Wordlist
        # Cocatenate all items in Wordlists as string with spaces between items
        Text1 = " ".join(Wordlist1)
        Text2 = " ".join(Wordlist2)
        # Append Text1 to cleanlist
        cleanlist.append(Text1)
        # Append the Text2 to stemlist
        stemlist.append(Text2)

    pass


#%%%% Stem ScriptDialogue and Main4Dialogue using Stemmer

# create empty lists for Stemmer function
StandupClean = []
StandupStem = []


# Run Stemmer
Stemmer(StandupDialogue, StandupStem, StandupClean)


# Check Lists
print(len(StandupClean))  # 191 (expected rows for df)
print(len(StandupStem))  # 191 (expected rows for df)

# place into data frames
StandUp = pd.DataFrame()
StandUp['Dialogue'] = StandupClean

StandUpStem = pd.DataFrame()
StandUpStem['Dialogue'] = StandupStem


#%%  Save StandUp and StandUpStem to csv

StandUp.to_csv(
    '/Users/Juan/Desktop/Python/Spyder Projects/IST 736 Projects/Group Project/data/StandUp.csv',
    index=False)

StandUpStem.to_csv(
    '/Users/Juan/Desktop/Python/Spyder Projects/IST 736 Projects/Group Project/data/StandUpStem.csv',
    index=False)

#%%################### GET DATA ########################

"""
Will be using NewsAPI.org to get finance and sports news as data sources.

Will use both methods (request module and build url tuple) shown in class
to extract the data from NewsAPI.org
"""
### Request Module method for finance news

# Get top stories on forbes.com for Feb 25,2021
BaseURL = "https://newsapi.org/v2/everything"  # base url for newsapi

# defined parameters for query
FinParams = {'apiKey': 'a2f3..replace with your api..4a217',
             'domains': 'forbes.com',
             'pageSize': '95',
             'from': '2021-02-25',
             'to': '2021-02-25',
             'sortBy': 'top'}

# get the api data. Save as finance
finance = requests.get(BaseURL, FinParams)


### Build url tuple method for science articles

# get top stories from smithsonianmag.com
URL = ('https://newsapi.org/v2/everything?'
       'apiKey=a2f3..replace with your api..4a217&'
       'domains=smithsonianmag.com&'
       'pageSize=95&'
       'sortBy=top')

# get the api data. Save as sports
science = requests.get(URL)


#%% Save text in json format. Check data by print

# print the financetxt
financetxt = finance.json()
print(financetxt)

#print the sportstxt
sciencetxt = science.json()
print(sciencetxt, "\n")

#%% Cleaning the Data

"""
the title,description and content of the articles will be extracted. 
All three parameters will be concatenated by article
"""

# Clean the financetxt

FinContent = []  # empty list

for item in financetxt["articles"]:  # access list of dictionaries under articles
    Title = item["title"]  # extract title
    Title = re.sub('\d', "", Title)  # remove digits
    Title = re.sub('\$', "", Title)  # remove dollar signs
    Title = re.sub(',', "", Title)  # remove commas

    Descript = item["description"]  # extract description
    Descript = re.sub('\d', "", Descript)  # remove digits
    Descript = re.sub('\$', "", Descript)  # remove dollar signs
    Descript = re.sub(',', "", Descript)  # remove commas

    Content = item['content']  # extract content
    Content = re.sub('\$', "", Content)  # remove dollar signs
    """
    Each content line ends with a 'word... [+digit chars]' 
    which will be removed.
    """
    Content = re.sub("\[", "", Content)  # remove '['
    Content = re.sub('\+', "", Content)  # remove '+'
    Content = re.sub('\d', "", Content)  # remove digits
    Content = re.sub('chars', "", Content)  # remove 'chars'
    Content = re.sub('\]', "", Content)  # remove ']'
    Content = re.sub('Getty Images', "", Content)  # remove ref to getty
    Content = re.sub('getty', "", Content)  # remove ref to getty
    Content = re.sub("...$", "", Content)  # remove '...' at end of string
    # the last sub  created incomplete words that will be cleaned by
    # removing the 2 to 20 letters at end of string.
    Content = re.sub('[A-Za-z]{2,20}$', "", Content)
    Content = re.sub(',', "", Content)

    # add concatenated Title, Descript and Content to FinContent
    FinContent.append(Title + " " + Descript + " " + Content)


#%% Clean sciencetxt duplicating method above.

ScienceContent = []


for item in sciencetxt['articles']:
    title = item['title']
    title = re.sub('\d', "", title)
    title = re.sub('\$', "", title)
    #title = re.sub(',', "", title)

    descript = item['description']
    descript = re.sub('\d', "", descript)
    descript = re.sub('\$', "", descript)
    descript = re.sub(',', "", descript)

    content = item['content']
    content = re.sub('\$', "", str(content))
    content = re.sub(',', "", content)
    content = re.sub('\[', "", content)
    content = re.sub('\+', "", content)
    content = re.sub('\d', "", content)
    content = re.sub('chars', "", content)
    content = re.sub('\]', "", content)
    content = re.sub('...$', "", content)
    content = re.sub(r'[A-Za-z]{2,20}$', "", content)

    ScienceContent.append(title + " " + descript + " " + content)


#%% place FinContent/ScienceContent in data frames and add "NotFunny" label

FinDF = pd.DataFrame()
FinDF['Content'] = FinContent
FinDF['Label'] = "NotFunny"

ScienceDF = pd.DataFrame()
ScienceDF['Content'] = ScienceContent
ScienceDF['Label'] = "NotFunny"

# Copy Standup DF as FunnyorNot and add label as "Funny"

FunnyorNot = StandUp.copy()
FunnyorNot = FunnyorNot.rename(columns={'Dialogue': 'Content'})
FunnyorNot['Label'] = "Funny"

# Merge the FinDF and ScienceDF dataframes into Funny_or_Not
FunnyorNot = pd.merge(FunnyorNot, FinDF, how='outer')
FunnyorNot = pd.merge(FunnyorNot, ScienceDF, how='outer')


#%% Save FunnyorNot to csv

FunnyorNot.to_csv(
    '/Users/Juan/Desktop/Python/Spyder Projects/IST 736 Projects/Group Project/data/FunnyorNot.csv',
    index=False)

#%%
