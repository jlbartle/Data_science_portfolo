# FDA adverse drug events analysis

# imports
import pandas as pd
import numpy as np
import json
import simplejson
from operator import itemgetter
import datetime as datetime
import requests
from flatten import flatten
import dask.dataframe as dd
import time

#sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# visualization imports
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


# read in consolidated df created in
t=time.time()

pathToData = 'X://consolidatedData.csv'

df_fda = dd.read_csv(pathToData,dtype={'authoritynumb': 'object',
       'patient.patientagegroup': 'float64',
       'patient.patientonsetage': 'float64',
       'patient.patientonsetageunit': 'float64',
       'patient.patientsex': 'float64',
       'seriousnesshospitalization': 'float64',
       'safetyreportid': 'object'})
df_fda=df_fda.compute()


df_fda = df_fda.drop(columns = ['authoritynumb', 'companynumb','reportduplicate.duplicatesource',
                                'patient.patientonsetageunit','patient.patientagegroup','primarysourcecountry','V1','index', 'receiver'])

# filter out all 2 and 3's from primarysource.qualification
# df_fda = df_fda[df_fda.primarysource.qualification != 1]

# drop health professionals
#df_fda = df_fda.drop(columns = 'primarysource.qualification')

# replace nan ages with -1
df_fda['patient.patientonsetage'] = df_fda['patient.patientonsetage'].fillna(-1)
# filter ages drop every age over 115 (probably reporting anomalies)
df_fda = df_fda.loc[df_fda['patient.patientonsetage'] < 115]
# convert ages to ints
df_fda['patient.patientonsetage'] = df_fda['patient.patientonsetage'].astype(int)

# replace nan weights with -1
df_fda['patient.patientweight'] = df_fda['patient.patientweight'].fillna(-1)
# filter weights drop every weight over 635kg (the heaviest man ever) (probably reporting anomalies)
df_fda = df_fda.loc[df_fda['patient.patientweight'] < 635]
# converting to int where floats aren't needed
df_fda['patient.patientweight'] = df_fda['patient.patientweight'].astype(int)

# replace nan sex with -1
df_fda['patient.patientsex'] = df_fda['patient.patientsex'].fillna(-1)
# converting to int where floats aren't needed
df_fda['patient.patientsex'] = df_fda['patient.patientsex'].astype(int)

# convert potential bool values
# fill nan with 0
# big caveat with these conversions FALSE could indicate UNREPORTED data not just FALSE
df_fda[['seriousnesshospitalization', 'seriousnessother','seriousnesslifethreatening', 'seriousnessdeath',
       'seriousnesscongenitalanomali', 'seriousnessdisabling']] = \
       df_fda[['seriousnesshospitalization', 'seriousnessother','seriousnesslifethreatening', 'seriousnessdeath',
              'seriousnesscongenitalanomali', 'seriousnessdisabling']].fillna(0)

# convert 1/0 columns to bol
df_fda[['seriousnesshospitalization', 'seriousnessother','seriousnesslifethreatening', 'seriousnessdeath',
       'seriousnesscongenitalanomali', 'seriousnessdisabling']] = \
       df_fda[['seriousnesshospitalization', 'seriousnessother','seriousnesslifethreatening', 'seriousnessdeath',
              'seriousnesscongenitalanomali', 'seriousnessdisabling']].astype('bool')

# convert receipt date to datetime
df_fda['receiptdate'] = pd.to_datetime(df_fda['receiptdate'],format='%Y%m%d')
df_fda['receiptdate'] = df_fda['receiptdate'].dt.date