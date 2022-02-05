import nltk
import nltk.collocations

SLpath = "/Users/becky/Desktop/Corpus1910.txt"
f = open('Corpus1910.txt', encoding="utf8")
Corpus1910 = f.read()

len(Corpus1910)
12001

Corpus1951.lower()
mytokens = nltk.word_tokenize(Corpus1910)
print(mytokens)

from nltk import FreqDist
ndist = FreqDist(mytokens)
nitems = ndist.most_common(330)
for item in nitems:
    print (item[0], '\t', item[1])

SLpath = "/Users/becky/Desktop/Corpus1951.txt"
f = open('Corpus1951.txt', encoding="utf8")
Corpus1951 = f.read()

Corpus1951.lower()
mytokens = nltk.word_tokenize(Corpus1951)
print(mytokens)

len(Corpus1951)
37822

from nltk import FreqDist
ndist = FreqDist(mytokens)
nitems = ndist.most_common(330)
for item in nitems:
    print (item[0], '\t', item[1])

SLpath = "/Users/becky/Desktop/Corpus1961.txt"
f = open('Corpus1961.txt', encoding="utf8")
Corpus1961 = f.read()
Corpus1961.lower()
mytokens = nltk.word_tokenize(Corpus1961)
print(mytokens)
len(Corpus1961)
102116

from nltk import FreqDist
ndist = FreqDist(mytokens)
nitems = ndist.most_common(330)
for item in nitems:
    print (item[0], '\t', item[1])

SLpath = "/Users/becky/Desktop/Corpus1971.txt"
f = open('Corpus1971.txt', encoding="utf8")
Corpus1971 = f.read()
Corpus1971.lower()
len(Corpus1971)
174250
mytokens = nltk.word_tokenize(Corpus1971)
print(mytokens)
from nltk import FreqDist
ndist = FreqDist(mytokens)
nitems = ndist.most_common(330)
for item in nitems:
    print (item[0], '\t', item[1]) 

SLpath = "/Users/becky/Desktop/Corpus1981.txt"
f = open('Corpus1981.txt', encoding="utf8")
Corpus1981 = f.read()
Corpus1981.lower()
mytokens = nltk.word_tokenize(Corpus1981)
print(mytokens)

len(Corpus1981)
129191

from nltk import FreqDist
ndist = FreqDist(mytokens)
nitems = ndist.most_common(330)
for item in nitems:
    print (item[0], '\t', item[1])

SLpath = "/Users/becky/Desktop/Corpus1991.txt"
f = open('Corpus1991.txt', encoding="utf8")
Corpus1991 = f.read()
Corpus1991.lower()
mytokens = nltk.word_tokenize(Corpus1991)
print(mytokens)
len(Corpus1991)
820463
from nltk import FreqDist
ndist = FreqDist(mytokens)
nitems = ndist.most_common(330)
for item in nitems:
    print (item[0], '\t', item[1])

import uuid
import random
import logging
import pandas as pd
from confluent_kafka import Producer
import json
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

bootstrap_servers='127.0.0.1:9092'
topic='test'
msg_count=5

def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result.
        Triggered by poll() or flush(). """
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Message delivered to {}'.format(msg.topic()))        
def confluent_kafka_producer():
    p = Producer({'bootstrap.servers': bootstrap_servers})
    for data in simple_messages:        
        record_key = str(uuid.uuid4())
        record_value = json.dumps({'data': data})   
        p.produce(topic, key=record_key, value=record_value, on_delivery=delivery_report)
        p.poll(0)

    p.flush()
    print('we\'ve sent {count} messages to {brokers}'.format(count=len(simple_messages), brokers=bootstrap_servers))
    
confluent_kafka_producer()
logging.basicConfig(level=logging.DEBUG)
client = KSQLAPI(url='http://localhost:8088', timeout=60)

pip install ksql
pip install confluent_kafka

confluent start ksql-server
from ksql import KSQLAPI
client = KSQLAPI('http://ksql-server:8088')

import logging
from ksql import KSQLAPI
logging.basicConfig(level=logging.DEBUG)
client = KSQLAPI('http://ksql-server:8088')

Corpus_text = [
‘Enter sighting descriptions here'
]

bootstrap_servers='127.0.0.1:9092'
topic='test'
msg_count=5

def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result.
        Triggered by poll() or flush(). """
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Message delivered to {}'.format(msg.topic()))

def confluent_kafka_producer():
    p = Producer({'bootstrap.servers': bootstrap_servers})
    for data in simple_messages:
        record_key = str(uuid.uuid4())
        record_value = json.dumps({'data': data})     
        p.produce(topic, key=record_key, value=record_value, on_delivery=delivery_report)
        p.poll(0)
    p.flush()
    print('we\'ve sent {count} messages to {brokers}'.format(count=len(simple_messages), brokers=bootstrap_servers))
    
confluent_kafka_producer() 

logging.basicConfig(level=logging.DEBUG)
client = KSQLAPI(url='http://localhost:8088', timeout=60)

client.create_table(table_name='test_data',
                   columns_type=['data varchar'],
                   topic='test',
                   value_format='JSON',
                   key='data')
client.ksql('show tables')
res = client.query('select * from test_data limit 5')

def parse_results(res):
    res = ''.join(res)
    res = res.replace('\n', '')
    res = res.replace('}{', '},{')
    res = '[' + res + ']'
    return json.loads(res)

res_dict = parse_results(res)
def apply_sent(res):
    sent_res = []
    for r in res:
        sid = SentimentIntensityAnalyzer()
        try:
            sent_res.append(sid.polarity_scores(r['row']['columns'][2]))
        except TypeError:
            print('limit reached')
    return sent_res

send_res = apply_sent(res_dict)

send_res  


mybigrams = list(nltk.bigrams(mytokens))
print(mybigrams)
