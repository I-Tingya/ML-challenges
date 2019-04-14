#import all the libraries required for the project here

from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter

import operator
import pandas as pd 
import numpy as np 
import scipy
import matplotlib.pyplot as plt 
import warnings 
warnings.filterwarnings('ignore')

from gensim import corpora, models
import gensim
from gensim.models import CoherenceModel 
from wordcloud import WordCloud
from gensim.models import Word2Vec

from pyod.models.hbos import HBOS

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer