from libraries import *

#execute below only once
#nltk.download('punkt')
#nltk.download('stopwords')


stop_words=set(stopwords.words('english'))
porter_stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

with open(r"../data/training_data.txt",'r',encoding='utf-8') as file_:
    qry = file_.read().splitlines()


#function to process raw text and return tokens per sentence
def preprocess_(qry):
  #print('inside preprocess')
  tokens_re = []
  for item in qry:
    temp = tokenizer.tokenize(item)
    temp = ' '.join(temp)
    tokens_re.append(temp)

  tokens_gr = []
  for item in tokens_re:
    item = item.lower()
    temp = word_tokenize(item)
    tokens_gr.append(temp)

  
  filtered_tokens = []
  for item in tokens_gr:
      temp = [token for token in item if token not in stop_words]
      filtered_tokens.append(temp)

  stemmed_tokens = []
  for tokens in filtered_tokens:
      temp = [porter_stemmer.stem(token) for token in tokens]# if ' ' not in token]
      #temp1 = [token for token in tokens if ' ' in token]
      stemmed_tokens.append(temp)
  return stemmed_tokens

#call the function to clean the train messages
train_stem = preprocess_(qry)

#function which returns sorted tokens counts in entire corpus
def word_count(token_sents):
    #print('inside word count')
    tr_stem_com=[]
    for i in token_sents:
        tr_stem_com+=i
    tok_cnt = Counter(tr_stem_com)
    sorted_tok_count = sorted(tok_cnt.items(), key=operator.itemgetter(1),reverse=True)
    return sorted_tok_count

tok_cnt = word_count(train_stem)
word_cnt_ser = pd.Series(data=[v for k,v in tok_cnt],index = [k for k,v in tok_cnt] )


#word_cnt_ser.iloc[0:10].plot(kind='barh')
#plt.show()
#word_cnt_ser.iloc[(word_cnt_ser.size-20):].plot(kind='barh')
#plt.show()

#save the tokens in a text file for reuse
'''with open(r'../data/train_stem.txt','w',encoding='utf-8') as file_:
    for i in train_stem:
        print(' '.join(i),file=file_)'''

#ldamodel =  models.LdaModel.load(r'../trained_models/lda200/ldamodel200.model')
#dictionary = corpora.Dictionary(train_stem)



