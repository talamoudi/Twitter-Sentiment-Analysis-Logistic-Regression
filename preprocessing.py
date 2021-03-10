import os

import re
import string

import random

import nltk
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

from utils import write_file




def get_tweets(pos_path='positive_tweets.json',neg_path='negative_tweets.json'):
    positive_tweets = twitter_samples.strings(pos_path)
    negative_tweets = twitter_samples.strings(neg_path)
    return positive_tweets, negative_tweets





def process_tweets(tweets):
    processed_tweets = list()
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    st_words = stopwords.words('english')
    stemmer = PorterStemmer() 
    
    for tweet in tweets:
        new_tweet = tweet.replace('\n','')
        new_tweet = re.sub(r'^RT[\s]+', '', new_tweet)
        new_tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', new_tweet)
        new_tweet = re.sub(r'#', '', new_tweet)
        tweet_tokens = tokenizer.tokenize(new_tweet)
        new_tokens = list()
        for word in tweet_tokens:
            if word not in st_words and word not in string.punctuation:
                new_tokens.append( stemmer.stem(word) )
        if len(new_tokens) > 0:
            processed_tweets.append(new_tokens)

    return processed_tweets





if __name__ == "__main__":

    user = os.path.expanduser('~')
    if not os.path.isdir(user + '/nltk_data/corpora/twitter_samples'):
        nltk.download('twitter_samples')
    
    if not os.path.isdir(user + '/nltk_data/corpora/stopwords'):
        nltk.download('stopwords')

    if not os.path.isdir(user + '/nltk_data/tokenizers'):
        nltk.download('tokenizers')

    pos, neg = get_tweets()

    pos_tweets = process_tweets(pos)
    random.shuffle(pos_tweets)
    index = int(len(pos_tweets) * 0.1) + 1
    testing_data = pos_tweets[:index]
    training_data = pos_tweets[index:]
    training_labels = [1]*len(training_data)
    testing_labels = [1]*len(testing_data)

    neg_tweets = process_tweets(neg)
    random.shuffle(neg_tweets)
    index = int(len(neg_tweets) * 0.1) + 1
    testing_data.extend( neg_tweets[:index] )
    training_data.extend( neg_tweets[index:] )
    training_labels.extend([0]*len(neg_tweets[index:]))
    testing_labels.extend([0]*len(neg_tweets[:index]))

    assert(len(training_labels) == len(training_data))
    assert(len(testing_labels) == len(testing_data))

    write_file('./Data/training_data.csv', training_data)
    write_file('./Data/testing_data.csv', testing_data)
    write_file('./Data/training_labels.csv', training_labels)
    write_file('./Data/testing_labels.csv', testing_labels)