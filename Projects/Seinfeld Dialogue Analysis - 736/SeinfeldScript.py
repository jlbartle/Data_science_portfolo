#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:38:33 2021

Author:    Juan Ventosa
Course:    IST 736 Text Mining
Prof:      Dr. Ami Gates
Grp Proj:  Seinfeld Script
Date:      2/23/2021

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
import os
import string


#%%################### LOAD THE DATA ###############

# Create path to the data on local drive
Corpus = '/Users/Juan/Desktop/Python/Spyder Projects/IST 736 Projects/Group Project/data/corpus.txt'

# Read the Corpus and check first 10 lines of script
with open(Corpus) as content:
    script = content.readlines()

print(script[:10])

"""
Character and Dialogue are separated by ":" which can be used to split lines.
Note: there are also non-dialogue lines that need to be removed
"""

#%% Extract Character and Dialogue into lists

"""
Note: Some Characters and Dialogues contain action descriptions in parenthesis.
Example: Jerry (shudders): 'You don't say. (says while nodding head)
Since these action descriptions are not needed and actually taint the data, 
they have been removed during the creation of these lists
"""

# Create empty lists
Character = []
Dialogue = []


# Iterate through script
for row in script:
    # Search for rows with ":" (i.e. ignore all other lines)
    if re.search(":", row):
        if re.findall('^\(', row):  # if it begins with "("
            continue  # skip row
        if re.findall('^\[', row):  # if it begins with "["
            continue  # skip row
        if re.findall('^\*', row):  # if it begins with "*"
            continue  # skip row
        if not re.findall('^\%', row):  # if it doesn't begin with "%"
            # split the row at the first ":"
            # store row before split in 'speaker' and row after split in 'line'
            speaker, line = row.split(":", 1)

            # clean speaker by removing empty spaces
            speaker = re.sub(" ", "", speaker)
            # remove action descriptions in (parenthesis) for speaker
            if re.search("\(", speaker):  # if open parenthesis exist
                # place everything before "(" in clean_speaker
                clean_speaker, description = speaker.split("(", 1)
                # append Character list with clean_speaker
                Character.append(clean_speaker)
            else:
                Character.append(speaker)  

            """
            remove action descriptions in (parenthesis) from line
               Note: some lines contain two action descriptions
            """
            if re.search("\(", line):  # if open parenthesis exist
                # place before "(" in l1, after "(" in action
                l1, action = line.split("(", 1)
                if re.search("\)", action):  # if close parenthesis exist
                    # place before ")" in action, after ")"in l2
                    action, l2 = action.split(")", 1)
                    if re.search("\(", l2):  # if open parenthesis exist
                        # l2 before "(" and action after "("
                        l2, action2 = l2.split("(", 1)
                        if re.search("\)", action2):  # if close parenthesis exist
                            # action2 before ")", l3 after ")"
                            action2, l3 = action2.split(")", 1)
                            l = l1 + l2 + l3  # concatenate l1, l2 & l3
                            Dialogue.append(l)
                        else:  # if no close parenthesis in action2
                            l = l1 + l2  # concatenate l1 and l2
                            Dialogue.append(l)
                    else:  # if no open parenthesis in l2
                        l = l1 + l2  # concatenate l1 and l2
                        Dialogue.append(l)
                else:
                    Dialogue.append(l1)  # if no close parenthesis in action
            else:
                Dialogue.append(line)  # if no open parenthesis in line

# Check list
#print(Character[:5])  # print 1st 5 items in Character
#print(Dialogue[:5])  # print 1st 5 items in Dialogue  


#%% Place Characters and Dialogue in dataframe
ScriptDF = pd.DataFrame(columns=["Character", "Dialogue"])
ScriptDF["Character"] = Character
ScriptDF["Dialogue"] = Dialogue

# Get value counts of Characters
ScriptDF["Character"].value_counts()

"""
Value counts gives the number of times each value occurs in a column. For the
Character column it gives the number of times JERRY, GEORGE, ELAINE, KRAMER,
Newman, etc.
"""

#%% Plot the top 20 Characters with the most lines

# Copy the ScriptDF data frame to get chart data
LineCount = ScriptDF.copy()
# Group the data by Character
LineCount = LineCount.groupby(['Character'], as_index=False).count()
# Sort the data by number of Dialogue (a.k.a lines)
LineCount = LineCount.sort_values('Dialogue', ascending=False)
# Rename Dialogue column as 'Line_Count'
LineCount.columns = ['Character', 'Line_Count']
# Remove index
LineCount = LineCount.reset_index(drop=True)
# print first 20 lines
print(LineCount.iloc[:20])

# Plot the bar graph

plt.bar("Character", "Line_Count", data=LineCount.iloc[:20])
plt.title("The 20 Seinfeld Characters with The Most Lines")
plt.ylabel("Number of Lines")
plt.xlabel("Character Name")
plt.xticks(rotation=90, fontsize=10)
plt.savefig("The 20 Seinfeld Characters with the Most Lines")
plt.show()

#%% Add Label column to ScriptDF

"""
Main objective is to classify the main characters: Jerry, George, Elaine and 
Kramer.  Therefore Label will reflect main character if Character = a main
character.  All other characters will be labeled as "Other"
Note: Other includes and characters who say the same line simultaneously.
Example: Jerry and George: "Yeah."
"""

Label = []

for i in ScriptDF.Character:
    if i == 'JERRY':
        Label.append('JERRY')
        continue
    if i == 'GEORGE':
        Label.append('GEORGE')
        continue    
    if i == 'ELAINE':
        Label.append('ELAINE')
        continue
    if i == 'KRAMER':
        Label.append('KRAMER')
    else:
        Label.append('OTHER')

ScriptDF['Label'] = Label      

# Create separate data frame with only Jerry, George, Elaine and Kramer
Main4Script = ScriptDF.copy()
droplist = Main4Script.loc[Main4Script['Label'] == 'OTHER'].index
Main4Script = Main4Script.drop(droplist)


#%% Plot the number of dialogue lines by character

x_axis = ['JERRY', 'OTHER', 'GEORGE', 'ELAINE', 'KRAMER']
y_axis = ScriptDF['Label'].value_counts()
color = ["blue", "red", "gold", "orange", "green"]

plt.bar(x_axis, y_axis, color=color)
plt.title('Number of Lines by Character in Seinfeld')
plt.xlabel("Characters")
plt.ylabel("Number of Lines")
plt.savefig("Number of Lines by Character in Seinfeld")
plt.show()

#%% Save ScriptDF and Main4Script to csv files

ScriptDF.to_csv(
    '/Users/Juan/Desktop/Python/Spyder Projects/IST 736 Projects/Group Project/data/ScriptDF.csv',
    index=False) 
Main4Script.to_csv(
    '/Users/Juan/Desktop/Python/Spyder Projects/IST 736 Projects/Group Project/data/Main4Script.csv',
    index=False)

#%%#######################################
#        VECTORIZING THE DATA            #
##########################################

# Uncomment to reload csv into ScriptDF data frame
#ScriptDF = pd.read_csv(
#    '/Users/Juan/Desktop/Python/Spyder Projects/IST 736 Projects/Group Project/data/ScriptDF.csv')

# Create Empty lists
ScriptDialogue = []
ScriptLabels = []
ScriptCharacter = []

for line in ScriptDF['Dialogue']:
    ScriptDialogue.append(line)

for character in ScriptDF['Character']:
    ScriptCharacter.append(character)

for label in ScriptDF['Label']:
    ScriptLabels.append(label)

# Uncomment to Reload Main4Script
#Main4Script = pd.read_csv(
#    '/Users/Juan/Desktop/Python/Spyder Projects/IST 736 Projects/Group Project/data/Main4Script.csv')

# Create Empty lists
Main4Dialogue = []
Main4Labels = []
Main4Character = []

for line in Main4Script['Dialogue']:
    Main4Dialogue.append(line)

for character in Main4Script['Character']:
    Main4Character.append(character)

for label in Main4Script['Label']:
    Main4Labels.append(label)

#%%####### STEMMING THE DATA ##################

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

#%% Stem ScriptDialogue and Main4Dialogue using Stemmer


# create empty lists for Stemmer function
ScriptDialogueClean = []
ScriptDialogueStem = []
Main4DialogueClean = []
Main4DialogueStem = []

# Run Stemmer
Stemmer(ScriptDialogue, ScriptDialogueStem, ScriptDialogueClean)
Stemmer(Main4Dialogue, Main4DialogueStem, Main4DialogueClean)


# Check Lists
print(len(ScriptDialogueClean))  # 54084 (expected rows for df)
print(len(ScriptDialogueStem))  # 54084 (expected rows for df)

print(len(Main4DialogueClean))  # 39502 (expected rows for df)
print(len(Main4DialogueStem))  # 39502 (expected rows for df)

#%% Create custom stop words for stemmed data set

# Store the NLTK english stopwords
NLTKStopWords = stopwords.words('english')

# Create empty list for stemmed stop words
StemStop = []

# Loop through NLTKStopWords to stem each word and append to StemStop
for word in NLTKStopWords:
    stem = PS.stem(word)
    StemStop.append(stem)

# Check StemStop
print(StemStop)


#%%######### DEFINE THE VECTORIZERS ###############

# Define parameters for custom CountVectorizer
CV = CountVectorizer(input="content", 
                     stop_words=StemStop,  # custom defined
                     ngram_range=(1, 2),  # unigram & bigram
                     binary=False,  # frequency
                     analyzer='word',  # feature as words
                     min_df=5)

# Define parameters for custom TFidfVectorizer
Tfidl1 = TfidfVectorizer(input="content",
                         stop_words=StemStop,  # custom defined
                         ngram_range=(1, 2),  # unigram & bigram
                         norm='l1',  # document length normalization
                         analyzer='word',  # feature as words
                         min_df=5)

# Define parameters for custom TFidfVectorizer
Tfidl2 = TfidfVectorizer(input="content",
                         stop_words=StemStop,  # custom defined
                         ngram_range=(1, 2),  # unigram & bigram
                         norm='l2',  # document length normalization
                         analyzer='word',  # feature as words
                         min_df=5)


#%%%######## VECTORIZE SCRIPTSDF STEM DIALOGUE ##################

# vectorize for MultinomialNB Term Frequency
ScriptCV = CV.fit_transform(ScriptDialogueStem)

# vectorize for MultinomialNB Document length normalization
ScriptTVl1 = Tfidl1.fit_transform(ScriptDialogueStem)

# vectorize for MultinomialNB Euclidean length normalization
ScriptTVl2 = Tfidl2.fit_transform(ScriptDialogueStem)

""" Vocabulary for all four vectorizers should be identical """

# Check vocabulary lengths
print(len(CV.vocabulary_), len(Tfidl1.vocabulary_),
      len(Tfidl2.vocabulary_))
# Output: 8724, 8724, 8724


#%% Build four data frames one for each vectorizer value type

Features = CV.get_feature_names()  # get features for df

# Place the document matrices into data frames using Features as columns
ScriptCVDF = pd.DataFrame(ScriptCV.toarray(), columns=Features)
ScriptTVl1DF = pd.DataFrame(ScriptTVl1.toarray(), columns=Features)
ScriptTVl2DF = pd.DataFrame(ScriptTVl2.toarray(), columns=Features)

# add the labels to the data frames
ScriptCVDF['Label'] = ScriptLabels
ScriptTVl1DF['Label'] = ScriptLabels
ScriptTVl2DF['Label'] = ScriptLabels

# check the data frames
print(ScriptCVDF.head(3))
print(ScriptTVl1DF.head(3))
print(ScriptTVl2DF.head(3))


#%%%######## VECTORIZE MAIN4 STEM DIALOGUE ##########

# vectorize for MultinomialNB Term Frequency
Main4CV = CV.fit_transform(Main4DialogueStem)

# vectorize for MultinomialNB Document length normalization
Main4TVl1 = Tfidl1.fit_transform(Main4DialogueStem)

# vectorize for MultinomialNB Euclidean length normalization
Main4TVl2 = Tfidl2.fit_transform(Main4DialogueStem)

""" Vocabulary for all four vectorizers should be identical """

# Check vocabulary lengths
print(len(CV.vocabulary_), len(Tfidl1.vocabulary_),
      len(Tfidl2.vocabulary_))
# Output: 6605, 6605, 6605, 6605


#%% Build four data frames one for each vectorizer value type

Features = CV.get_feature_names()  # get features for df

# Place the document matrices into data frames using Features as columns
Main4CVDF = pd.DataFrame(Main4CV.toarray(), columns=Features)
Main4TVl1DF = pd.DataFrame(Main4TVl1.toarray(), columns=Features)
Main4TVl2DF = pd.DataFrame(Main4TVl2.toarray(), columns=Features)

# add the labels to the data frames
Main4CVDF['Label'] = Main4Labels
Main4TVl1DF['Label'] = Main4Labels
Main4TVl2DF['Label'] = Main4Labels

# check the data frames
print(Main4CVDF.head(3))
print(Main4TVl1DF.head(3))
print(Main4TVl2DF.head(3))

#%%
# Save dataframes as CSV files
#ScriptCVDF.to_csv(
#    '/Users/Juan/Desktop/Python/Spyder Projects/IST 736 Projects/Group Project/output/CVDF.csv',
#    index=False)
#ScriptTVl1DF.to_csv(
#    '/Users/Juan/Desktop/Python/Spyder Projects/IST 736 Projects/Group Project/output/TVl1.csv',
#    index=False)
#ScriptTVl2DF.to_csv(
#    '/Users/Juan/Desktop/Python/Spyder Projects/IST 736 Projects/Group Project/output/TVl2.csv',
#    index=False)

#%%###############################################
#               MODEL ANALYSIS                   #
##################################################

"""
Created a model for each data set to be trained and tested. This was done in
order to insure the feature_prob_log for each model are saved separately for
accurate retrieval
"""

# Define the models
S1MNBclf = MultinomialNB(alpha=1.0,  # apply Laplace smoothing
                         fit_prior=True)  # learn class prior probabilities

S2MNBclf = MultinomialNB(alpha=1.0,  # apply Laplace smoothing
                         fit_prior=True)  # learn class prior probabilities

S3MNBclf = MultinomialNB(alpha=1.0,  # apply Laplace smoothing
                         fit_prior=True)  # learn class prior probabilities

M4aMNBclf = MultinomialNB(alpha=1.0,  # apply Laplace smoothing
                          fit_prior=True)  # learn class prior probabilities

M4bMNBclf = MultinomialNB(alpha=1.0,  # apply Laplace smoothing
                          fit_prior=True)  # learn class prior probabilities

M4cMNBclf = MultinomialNB(alpha=1.0,  # apply Laplace smoothing
                          fit_prior=True)  # learn class prior probabilities


#%% Split the data sets into train and test-validation sets

# Use a split of 75% Train and 25% Test-validation
np.random.seed(23)
STrainCV, STestCV = train_test_split(ScriptCVDF,
                                     train_size=2000, test_size=500)
STrainTVl1, STestTVl1 = train_test_split(ScriptTVl1DF,
                                         train_size=2000, test_size=500)
STrainTVl2, STestTVl2 = train_test_split(ScriptTVl2DF, 
                                         train_size=2000, test_size=500)
M4TrainCV, M4TestCV = train_test_split(Main4CVDF,
                                       train_size=2000, test_size=500)
M4TrainTVl1, M4TestTVl1 = train_test_split(Main4TVl1DF,
                                           train_size=2000, test_size=500)
M4TrainTVl2, M4TestTVl2 = train_test_split(Main4TVl2DF, 
                                           train_size=2000, test_size=500)
# Store the test labels
STestCVLabels = STestCV["Label"]
STestTVl1Labels = STestTVl1['Label']
STestTVl2Labels = STestTVl2['Label']
M4TestCVLabels = M4TestCV["Label"]
M4TestTVl1Labels = M4TestTVl1['Label']
M4TestTVl2Labels = M4TestTVl2['Label']

# Drop labels from test data sets
STestCV = STestCV.drop(["Label"], axis=1)
STestTVl1 = STestTVl1.drop(['Label'], axis=1)
STestTVl2 = STestTVl2.drop(['Label'], axis=1)
M4TestCV = M4TestCV.drop(["Label"], axis=1)
M4TestTVl1 = M4TestTVl1.drop(['Label'], axis=1)
M4TestTVl2 = M4TestTVl2.drop(['Label'], axis=1)

#%%% Define a non-replacement cross validation function

"""
Function wil require parameters: 
    DF - train data in df format
    model - classification model
    folds - number of cross validations
    3 empty lists:
        trainerror - record train accuracy
        testerror - record test accuracy
        testreport - record test classification reports as list of dictionaries
"""


def XFoldCV(DF, model, folds, trainerror, testerror, testreport):
    count = 1  # only needed if Confusion Matrix is uncommented
    index = 0  # starting index for spliting data frame
    index2 = index + int(len(DF) * ((100 / folds) * .01))  # first cutoff

    for i in range(folds):  # repeat process below folds times

        # split the data into train and test
        Train = DF.copy()  # copy the data frame
        Train = Train.reset_index(drop=True)  # reset indices
        # Take % of Train determined by fold- defined by index to index2 range
        Test = Train.iloc[index:index2, ] 
        # Remove the rows used for Test from Train
        for n in range(index, index2):
            Train = Train.drop(n)
        TrainLabel = Train["Label"]  # store Train labels
        TestLabel = Test["Label"]  # store Test labels
        Train = Train.drop(["Label"], axis=1)  # drop label col from Train
        Test = Test.drop(["Label"], axis=1)  # drop label col from Test

        Model = model.fit(Train, TrainLabel)  # Train the model
        PredictTrain = model.predict(Train)  # model predict train 
        PredictTest = model.predict(Test)  # model predict test
        # predicted train and test to check for overfitting

        #get F1 measure for Train and Test and classification report as dict
        Error1 = accuracy_score(TrainLabel, PredictTrain)
        Error2 = accuracy_score(TestLabel, PredictTest)
        Report = classification_report(
            TestLabel, PredictTest, output_dict=True)

        trainerror.append(Error1)  # append Train F1 measure to list
        testerror.append(Error2)  # append Test F1 measure to list
        testreport.append(Report)  # append report dictionary to list

        # Uncomment to plot a confusion matrix for each cross validation test
        #CM = ConfusionMatrixDisplay(
        #    confusion_matrix(TestLabel, PredictTest)).plot()
        #if count > 0:
        # title each iteration
        # plt.title("Sentiment Confusion Matrix" + str(count)) 
        # save each iteration.png
        # plt.savefig("Sentiment Confusion Matrix" + str(count))
        # plt.show  # print plot
        # count += 1
        # add folds % of DF to index
        index += int(len(DF) * ((100 / folds) * .01))
        # add folds % of DF to index2
        index2 += int(len(DF) * ((100 / folds) * .01))

    pass  # no return

#%% Perform a X-fold cross validation on the each of the Training data


# Empty lists to capture results for TrainCV cross validation
SCVTrainError = []
SCVTestError = []
SCVTestReport = []
M4CVTrainError = []
M4CVTestError = []
M4CVTestReport = []

# Empty lists to capture results for TrainTVl1 cross validation
STVl1TrainError = []
STVl1TestError = []
STVl1TestReport = []
M4TVl1TrainError = []
M4TVl1TestError = []
M4TVl1TestReport = []

# Empty lists to capture results for TrainTVl2 cross validation
STVl2TrainError = []
STVl2TestError = []
STVl2TestReport = []
M4TVl2TrainError = []
M4TVl2TestError = []
M4TVl2TestReport = []

# Apply XFoldCV function to all 3 Training data w/ respective models
XFoldCV(STrainCV, S1MNBclf, 5, SCVTrainError, SCVTestError, SCVTestReport)
XFoldCV(STrainTVl1, S2MNBclf, 5, STVl1TrainError,
        STVl1TestError, STVl1TestReport)
XFoldCV(STrainTVl2, S3MNBclf, 5, STVl2TrainError,
        STVl2TestError, STVl2TestReport)
XFoldCV(M4TrainCV, M4aMNBclf, 5, M4CVTrainError, M4CVTestError, M4CVTestReport)
XFoldCV(M4TrainTVl1, M4bMNBclf, 5, M4TVl1TrainError,
        M4TVl1TestError, M4TVl1TestReport)
XFoldCV(M4TrainTVl2, M4cMNBclf, 5, M4TVl2TrainError,
        M4TVl2TestError, M4TVl2TestReport)

#%% Create function to plot results

"""
This function will take in the TrainError, TestError and TestReport lists 
created in XFoldCV and use them to plot bar graphs of the results for:
    Train vs. Test Accuracy Scores
    Finance vs. Science Precision Scores
    Finance vs. Science Recall Scores
    Finance vs. Science F1-Scores
"""


def PlotResults(trainerror, testerror, testreport):

    index = 0  # used to help iterate through dictionary

    # Empty lists to collect and hold precision results by class
    JerryPrecision = []
    GeorgePrecision = []
    ElainePrecision = []
    KramerPrecision = []
    OtherPrecision = []

    # Empty lists to collect and hold recall results by class
    JerryRecall = []
    GeorgeRecall = []
    ElaineRecall = []
    KramerRecall = []
    OtherRecall = []

    # Empty lists to collect and hold F1-scores by class
    JerryF1 = []
    GeorgeF1 = []
    ElaineF1 = []
    KramerF1 = []
    OtherF1 = []

    """
    testreport is a list of dictionaries containining classification report 
    results for each class precision, recall and f1-score. Loop
    below will iterate testreport to append each score in the appropriate
    empty lists created above.
    """

    for n in range(0, len(testreport)):

        # append precisions to appropriate lists    
        JP = testreport[index]['JERRY']['precision']
        JerryPrecision.append(round(JP, 2))

        GP = testreport[index]['GEORGE']['precision']
        GeorgePrecision.append(round(GP, 2))

        EP = testreport[index]['ELAINE']['precision']
        ElainePrecision.append(round(EP, 2))

        KP = testreport[index]['KRAMER']['precision']
        KramerPrecision.append(round(KP, 2))

        # append recall to appropriate lists
        JR = testreport[index]['JERRY']['recall']
        JerryRecall.append(round(JR, 2))

        GR = testreport[index]['GEORGE']['recall']
        GeorgeRecall.append(round(GR, 2))

        ER = testreport[index]['ELAINE']['recall']
        ElaineRecall.append(round(ER, 2))

        KR = testreport[index]['KRAMER']['recall']
        KramerRecall.append(round(KR, 2))

        # append f1-scores to appropriate lists
        JF = testreport[index]['JERRY']['f1-score']
        JerryF1.append(round(JF, 2))

        GF = testreport[index]['GEORGE']['f1-score']
        GeorgeF1.append(round(GF, 2))

        EF = testreport[index]['ELAINE']['f1-score']
        ElaineF1.append(round(EF, 2))

        KF = testreport[index]['KRAMER']['f1-score']
        KramerF1.append(round(KF, 2))

        # if OTHER exists append scores to appropriate lists
        if len(testreport[index]) > 7:

            OP = testreport[index]['OTHER']['precision']
            OtherPrecision.append(round(OP, 2))

            OR = testreport[index]['OTHER']['recall']
            OtherRecall.append(round(OR, 2))

            OF = testreport[index]['OTHER']['f1-score']
            OtherF1.append(round(OF, 2))

        index += 1  # add one to index to grab next dictionary entry

    ClassReport = pd.DataFrame()
    ClassReport['JERRY_Precision'] = JerryPrecision
    ClassReport['JERRY_Recall'] = JerryRecall
    ClassReport['JERRY_F1Score'] = JerryF1
    ClassReport['GEORGE_Precision'] = GeorgePrecision
    ClassReport['GEORGE_Recall'] = GeorgeRecall
    ClassReport['GEORGE_F1Score'] = GeorgeF1
    ClassReport['ELAINE_Precision'] = ElainePrecision
    ClassReport['ELAINE_Recall'] = ElaineRecall
    ClassReport['ELAINE_F1Score'] = ElaineF1
    ClassReport['KRAMER_Precision'] = KramerPrecision
    ClassReport['KRAMER_Recall'] = KramerRecall
    ClassReport['KRAMER_F1Score'] = KramerF1

    if len(OtherPrecision) > 0:

        ClassReport['OTHER_Precision'] = OtherPrecision
        ClassReport['OTHER_Recall'] = OtherRecall
        ClassReport['OTHER_F1Score'] = OtherF1

    ClassReport = ClassReport.transpose()
    columnNames = ["Test " + str(i) for i in range(1, len(testreport) + 1)]
    ClassReport.columns = columnNames

    ReportTitle = input("Give ClassReport Title: ")
    ClassReport.to_csv(ReportTitle + ".csv", index=False)

    # Allows user input to add prefix to chart titles
    ChartTitle = input("Give Model Title: ") 

    x_axis = []
    count = 1

    for i in range(0, len(testreport)):
        x_axis.append('T' + str(count))
        count += 1

    plt.bar(x_axis, JerryF1, width=0.8, color='darkblue', label="JERRY")
    plt.title(ChartTitle + ' Jerry F1-Scores by Test Fold')
    plt.xlabel("Test Folds")
    plt.ylabel("F1-Scores")
    plt.show()

    plt.bar(x_axis, GeorgeF1, width=0.8, color='orange', label="GEORGE")
    plt.title(ChartTitle + " George F1-Scores by Test Fold")
    plt.xlabel("Test Folds")
    plt.ylabel("F1-Scores")
    plt.show()

    plt.bar(x_axis, ElaineF1, width=0.8, color='darkred', label="ELAINE")
    plt.title(ChartTitle + ' Elaine F1-Scores by Test Fold')
    plt.xlabel("Test Folds")
    plt.ylabel("F1-Scores")
    plt.show()

    plt.bar(x_axis, KramerF1, width=0.8, color='darkgreen', label="KRAMER")
    plt.title(ChartTitle + ' Kramer F1 Scores by Test Fold')
    plt.xlabel("Test Folds")
    plt.ylabel("F1-Scores")
    plt.show()
    #ax.bar(ind+(width*4), y_axis5, width, color='darkred',
    #       label="OTHER")

    plt.savefig(ChartTitle + ' Seinfeld F1-Scores by Class')
    plt.show()

    # Chart results of Train vs. Test Accuracy Scores
    y_axis6 = trainerror
    y_axis7 = testerror
    plt.title(ChartTitle + 'Train vs. Test Accuracy Scores')
    plt.xlabel('Test Folds')
    plt.ylabel('Accuracy Scores')
    plt.bar(x_axis, y_axis6, color='darkblue',
            label="Train", width=-0.4, align='edge')
    plt.bar(x_axis, y_axis7, color='darkred',
            label='Test', width=0.4, align='edge')
    plt.legend(bbox_to_anchor=(1.25, 1), loc='upper right')
    plt.savefig(ChartTitle + ' Train vs Test Accuracy Scores')
    plt.show()

    return print(ClassReport)


#%% Check the cross validation results the TrainCV data

print('Avg. Train accuracy = ', round(
    sum(SCVTrainError) / len(SCVTrainError), 2), '\n')
print('Avg. Test accuracy = ', round(sum(SCVTestError) / len(SCVTestError), 2))

PlotResults(SCVTrainError, SCVTestError, SCVTestReport)


#%% Check the cross validation results the TrainTVl1 data

print('Avg. Train accuracy = ', round(
    sum(STVl1TrainError) / len(STVl1TrainError), 2), '\n')
print('Avg. Test accuracy = ', round(
    sum(STVl1TestError) / len(STVl1TestError), 2))

PlotResults(STVl1TrainError, STVl1TestError, STVl1TestReport)


#%% Check the cross validation results the TrainTVl2 data

print('Avg. Train accuracy = ', round(
    sum(STVl2TrainError) / len(STVl2TrainError), 2), '\n')
print('Avg. Test accuracy = ', round(
    sum(STVl2TestError) / len(STVl2TestError), 2))

PlotResults(STVl2TrainError, STVl2TestError, STVl2TestReport)


#%% Check the cross validation results the M4TrainCV data

print('Avg. Train accuracy = ', round(
    sum(M4CVTrainError) / len(M4CVTrainError), 2), '\n')
print('Avg. Test accuracy = ', round(
    sum(M4CVTestError) / len(M4CVTestError), 2))

PlotResults(M4CVTrainError, M4CVTestError, M4CVTestReport)


#%% Check the cross validation results the M4TrainTVl1 data

print('Avg. Train accuracy = ', round(
    sum(M4TVl1TrainError) / len(M4TVl1TrainError), 2), '\n')
print('Avg. Test accuracy = ', round(
    sum(M4TVl1TestError) / len(M4TVl1TestError), 2))

PlotResults(M4TVl1TrainError, M4TVl1TestError, M4TVl1TestReport)


#%% Check the cross validation results the M4TrainTVl2 data

print('Avg. Train accuracy = ', round(
    sum(M4TVl2TrainError) / len(M4TVl2TrainError), 2), '\n')
print('Avg. Test accuracy = ', round(
    sum(M4TVl2TestError) / len(M4TVl2TestError), 2))

PlotResults(M4TVl2TrainError, M4TVl2TestError, M4TVl2TestReport)


#%% Train Models on entire Train data sets and test against Test-Validations

"""
The models performed relatively well on the Binary, Term Frequency and 
Euclidean data sets during cross validation.  The models will be trained on 
the entire train data sets for these three and use their respective 
Test-Validation holdouts as a final test to compare against their respective
cross validation results.

Note: Due to the models poor performance on the Document length normalized
data, this data set will be excluded from further analysis
"""
# Store the train labels
STrainCVLabels = STrainCV["Label"]
STrainTVl2Labels = STrainTVl2['Label']

M4TrainCVLabels = M4TrainCV["Label"]
M4TrainTVl2Labels = M4TrainTVl2['Label']


# Drop labels from train data sets
STrainCV = STrainCV.drop(["Label"], axis=1)
STrainTVl2 = STrainTVl2.drop(['Label'], axis=1)

M4TrainCV = M4TrainCV.drop(["Label"], axis=1)
M4TrainTVl2 = M4TrainTVl2.drop(['Label'], axis=1)


# train Multinomial model on TermFreq data and test it
SCV_MNB = S1MNBclf.fit(STrainCV, STrainCVLabels)
SCV_Predict = SCV_MNB.predict(STestCV)

M4CV_MNB = M4aMNBclf.fit(M4TrainCV, M4TrainCVLabels)
M4CV_Predict = M4CV_MNB.predict(M4TestCV)

# train Multinomila model on Euclidean normalized data and test it
STVl2_MNB = S3MNBclf.fit(STrainTVl2, STrainTVl2Labels)
STVl2_Predict = STVl2_MNB.predict(STestTVl2)

M4TVl2_MNB = M4cMNBclf.fit(M4TrainTVl2, M4TrainTVl2Labels)
M4TVl2_Predict = M4TVl2_MNB.predict(M4TestTVl2)

# Check Classes for index positions
print(SCV_MNB.classes_)      # [0]ELAINE [1]GEORGE [2]JERRY [3]KRAMER [4]OTHER
print(M4CV_MNB.classes_)     # [0]ELAINE [1]GEORGE [2]JERRY [3]KRAMER
print(STVl2_MNB.classes_)    # [0]ELAINE [1]GEORGE [2]JERRY [3]KRAMER [4]OTHER
print(M4TVl2_MNB.classes_)   # [0]ELAINE [1]GEORGE [2]JERRY [3]KRAMER


#%% Bernoulli model accuracy on test validation

# Get classification scores
SCV_Score = classification_report(STestCVLabels, SCV_Predict)
print(SCV_Score)

""" 
Accuracy = 0.32 which was above test error avg and range in cross validation 
"""

# Plot a Confusion Matrix on Test Prediction
SCV_CM = ConfusionMatrixDisplay(
    confusion_matrix(STestCVLabels, SCV_Predict)).plot(cmap="Greys")
plt.title("Full Script TF_MNB Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(labels=["Elaine", "George", "Jerry", "Kramer", "Other"], 
           ticks=[0, 1, 2, 3, 4])
plt.yticks(labels=["Elaine", "George", "Jerry", "Kramer", "Other"], 
           ticks=[0, 1, 2, 3, 4])
plt.savefig("Full Script TF_MNB Confusion Matrix")
plt.show  # print plot


#%% Multinomial-Term Freq model accuracy on test validation

# Get classification scores
M4CV_Score = classification_report(M4TestCVLabels, M4CV_Predict)
print(M4CV_Score)

""" 
Accuracy = 0.35 which was below test error range in cross validation 
"""

# Plot a Confusion Matrix on Test Prediction
M4CV_CM = ConfusionMatrixDisplay(
    confusion_matrix(M4TestCVLabels, M4CV_Predict)).plot(cmap="Blues")
plt.title("Main4 Script TF_MNB Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(labels=["Elaine", "George", "Jerry", "Kramer"], 
           ticks=[0, 1, 2, 3, ])
plt.yticks(labels=["Elaine", "George", "Jerry", "Kramer"], 
           ticks=[0, 1, 2, 3, ])
plt.savefig("Main4 Script TF_MNB Confusion Matrix")
plt.show  # print plot


#%% Multinomial-Euclidean model accuracy on test validation

# Get classification scores
STVl2_Score = classification_report(STestTVl2Labels, STVl2_Predict)
print(STVl2_Score)

""" Accuracy = 0.30 which was below the test error range """

# Plot a Confusion Matrix on Test Prediction
STVl2_CM = ConfusionMatrixDisplay(
    confusion_matrix(STestTVl2Labels, STVl2_Predict)).plot("YlOrRd")
plt.title("Full Script Euclidean MNB Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(labels=["Elaine", "George", "Jerry", "Kramer", "Other"], 
           ticks=[0, 1, 2, 3, 4])
plt.yticks(labels=["Elaine", "George", "Jerry", "Kramer", "Other"], 
           ticks=[0, 1, 2, 3, 4])
plt.savefig("Full Script Euclidean MNB Confusion Matrix")
plt.show  # print plot


#%% Multinomial-Euclidean model accuracy on test validation

# Get classification scores
M4TVl2_Score = classification_report(M4TestTVl2Labels, M4TVl2_Predict)
print(M4TVl2_Score)

""" Accuracy = 0.43 which was above the test error avg. and range """

# Plot a Confusion Matrix on Test Prediction
M4TVl2_CM = ConfusionMatrixDisplay(
    confusion_matrix(M4TestTVl2Labels, M4TVl2_Predict)).plot("YlOrRd")
plt.title("Main4 Script Euclidean MNB Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(labels=["Elaine", "George", "Jerry", "Kramer"], 
           ticks=[0, 1, 2, 3, ])
plt.yticks(labels=["Elaine", "George", "Jerry", "Kramer"], 
           ticks=[0, 1, 2, 3, ])
plt.savefig("Main4 Script Euclidean MNB Confusion Matrix")
plt.show  # print plot

#%% Define a function to plot top x features of two classes

"""Function accepts vectorizer, model and two lists and number of features"""


def PlotTopFeatures(vectorizer, model, nfeatures):
    # create empty list to store feature probability given class 

    count = 0
    feature_prob = []

    for i in range(len(model.classes_)):
        feature_prob.append([])  # create a list in feature_prob for each class
        # loop through each feature log prob class 
        for p in model.feature_log_prob_[count]:
            # convert log of prob to probability %
            prob = (10**p) * 100
            # append it to appropriate list in feature_prob
            feature_prob[count].append(prob)
        count += 1

    # Get the feature names
    Features = vectorizer.get_feature_names() 
    classlist = []  # empty list to store probability and features
    Prob = []  # empty list to store top feature probabilities
    Feat = []  # empty list to store top feature names
    classnames = []  # empty list to store classnames for plot
    TitleSuffix = input("Enter Model Type: ") 

    for i in model.classes_:  # append classes to classnames
        classnames.append(i)
    # Combine the feature probabilities given class with their feature names
    # then sort data in ascending order and take the last 10 (highest) values.
    n = 0

    for i in range(len(feature_prob)):
        classlist.append([])  # create list in classlist for each class
        # combine feature with probabilities & take top 10
        top10_feat_class = sorted(zip(feature_prob[n], Features))[-nfeatures:]
        # append to appropriate list in classlist
        classlist[n].append(top10_feat_class)
        n += 1

    num = 0  # a count value for loop
    # list of colors for graphs
    color = ['orange', 'gold', 'darkblue', 'darkgreen', 'darkred']

    # get the top probabilities and features by class to plot
    for i in range(len(classlist)):
        ProbFeat = classlist[num]
        for x in ProbFeat:
            for y in x:
                Prob.append(y[0])
                Feat.append(y[1])

        # plot features and probabilities for each class
        plt.barh(Feat, Prob, color=color[num], label=classnames[num])
        plt.ylabel("Features")
        plt.xlabel("Probability Percent")
        plt.title("Top " + str(nfeatures) + " " + classnames[num]
                  + " Features (" + TitleSuffix + ")")
        plt.savefig("Top " + str(nfeatures) + " " + classnames[num]
                    + " Features (" + TitleSuffix + ")")
        plt.show()

        Prob = []  # reset Prob list to empty
        Feat = []  # reset Feat list to empty
        num += 1  # add one to move to next class

    pass  # no return


#%% Use PlotTopFeatures to capture Top 10 Features of each class

# Note: CV was original vectorizer used for SCV_MNB
PlotTopFeatures(CV, SCV_MNB, 10)

#%% Note: CV was original vectorizer used for M4CV_MNB

PlotTopFeatures(CV, M4CV_MNB, 10)

#%% Note: Tfidl2 was original vectorizer used for STVl2_MNB

PlotTopFeatures(Tfidl2, STVl2_MNB, 10)  
#%% Note: Tfidl2 was original vectorizer used for M4TVl2_MNB

PlotTopFeatures(Tfidl2, M4TVl2_MNB, 10)   

#%%
