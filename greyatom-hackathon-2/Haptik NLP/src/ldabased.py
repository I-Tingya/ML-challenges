from libraries import *
from preprocess import preprocess_


#load the train file tokens
with open(r'../data/train_stem.txt','r',encoding='utf-8') as file_:
    train_stemm = file_.read().splitlines()
train_stem=[]
for i in train_stemm:
    train_stem.append(i.split())

#function which creates lda model/loads the trained model, returns model,corpus and dictionary
def lda_model():
    #load pre-trained model
    ldamodel =  models.LdaModel.load(r'../trained_models/lda200colab/ldamodel200colab.model')
    dictionary = corpora.Dictionary(train_stem)
    #print(dictionary)
    corpus = [dictionary.doc2bow(text) for text in train_stem]

    #keep below line only in case of training new model
    #ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=7, id2word = dictionary, passes=200)

    return ldamodel,corpus,dictionary

#call function 
ldamodel,corpus,dictionary = lda_model()

#execute below to save trained model
#ldamodel.save('ldamodel.model')

#print the topics and top 8 words of each topic
for topic in ldamodel.print_topics(num_topics=7, num_words=8):
    print(topic)

#below script to generate wordcloud showing 20 words in each topic 
'''num_topics = 7
topic_words = []
for i in range(num_topics):
    tt = ldamodel.get_topic_terms(i,20)
    topic_words.append([dictionary[pair[0]] for pair in tt])

    wordcloud = WordCloud().generate(' '.join(topic_words[i]))
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()'''


#read test file, tokenize it and create bow
test = pd.read_csv(r'../data/eval_data.csv')
test_msgs = test.Message.tolist()
stemmed_test_tokens = preprocess_(test_msgs)
corpus_ = [dictionary.doc2bow(text) for text in stemmed_test_tokens]

y_pred=[]
y_pred_baseline=[]

#for each sentence in test file predict the topic by trained LDA model
#in our case topic 2,4 and 6 seemed to be outlier so any sentence which fall under these 3
#were tagged as a outlier
for text in corpus_:
  topics = ldamodel[text]
  #print(topics)
  temp = [prob for top,prob in topics]
  idx = np.array(temp).argmax()
  temp = True if idx in [2,4,5] else False
  y_pred.append(temp)
  y_pred_baseline.append(True)

y_test=test['Outlier?'].tolist()

#scoring metrics if we naivly said all given test sentences are outliers
acc_score = metrics.accuracy_score(y_test,y_pred_baseline)
print('\nnaive baseline accuracy score ',acc_score)
prec_score = metrics.precision_score(y_test,y_pred_baseline)
print('\nnaive baseline precision score ',prec_score)

#model scores
acc_score = metrics.accuracy_score(y_test,y_pred)
print('\nmodel accuracy score ',acc_score)
prec_score = metrics.precision_score(y_test,y_pred)
print('\nmodel precision score ',prec_score)


#below code predicts the topic of each train sentence 
topic_train=[]

for text in corpus:
  topics = ldamodel[text]
  #print(topics)
  temp = [prob for top,prob in topics]
  idx = np.array(temp).argmax()
  topic_train.append(idx)

#se = pd.Series(topic_train)
#print(se.value_counts())
#create a data frame which stores train sentences and predicted topics in csv file
train_stem_temp = [' '.join(i) for i in train_stem]
topic_df = pd.DataFrame({'text':train_stem_temp,'topic':topic_train})
topic_df.to_csv(r'../data/generated/topic_df.csv',index=False)

#below code to find coherence score and model perplexity, both should be higher

'''# Compute Coherence Score, debug the error
coherence_model_lda = CoherenceModel(model=ldamodel, 
texts=train_stem, dictionary=dictionary, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)'''

# Compute Perplexity
print('\nPerplexity: ', ldamodel.log_perplexity(corpus))