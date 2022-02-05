# Textmining Naive Bayes Example
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
import os
from pathlib import Path
import numpy as np


raw_sf = open('E:\OneDrive\DataScience\IST Text Mining\-Project-\Seinfeld Scripts.csv')
df_sf = pd.read_csv(raw_sf)


# create directorys for corpus
Path('Seinfeld_corpus').mkdir(parents=True, exist_ok=True)
cwd = os.getcwd() + '\\Seinfeld_corpus\\'
os.chdir(cwd)
os.mkdir('JERRY')
os.mkdir('GEORGE')
os.mkdir('ELAINE')
os.mkdir('KRAMER')
os.mkdir('OTHER')

line_num = 0
for i in df_sf.values:
    if i[1] == 'JERRY':
        j = open(cwd + '\\JERRY\\j' + str(line_num) + '.txt', 'w+', encoding="utf8")
        j.write(str(i[2]) + '\n')
        j.close()
    elif i[1] == 'GEORGE':
        g = open(cwd + '\\GEORGE\\g' + str(line_num) + '.txt', 'w+', encoding="utf8")
        g.write(str(i[2]) + '\n')
        g.close()
    elif i[1] == 'ELAINE':
        e = open(cwd + '\\ELAINE\\e' + str(line_num) + '.txt', 'w+', encoding="utf8")
        e.write(str(i[2]) + '\n')
        e.close()
    elif i[1] == 'KRAMER':
        k = open(cwd + '\\KRAMER\\k' + str(line_num) + '.txt', 'w+', encoding="utf8")
        k.write(str(i[2]) + '\n')
        k.close()
    else:
        o = open(cwd + '\\OTHER\\o' + str(line_num) + '.txt', 'w+', encoding="utf8")
        o.write(str(i[2]) + '\n')
        o.close()
