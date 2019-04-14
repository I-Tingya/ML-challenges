from libraries import *
from preprocess import preprocess_

#load the train file tokens
with open(r'../data/train_stem.txt','r',encoding='utf-8') as file_:
    train_stemm = file_.read().splitlines()
train_stem=[]
for i in train_stemm:
    train_stem.append(i.split())

#make a single list of all the tokens and pass it to counter to get count of each token 
temp = []
for token in train_stem:
  temp = temp + token
count_dict = Counter(temp)

#below script creates dataframes for every topic containing sentences falling in that topic
'''topic_df = pd.read_csv(r'../data/generated/topic_df.csv')
print(topic_df.head())
for i in range(7):
    df = topic_df[topic_df['topic']==i]
    df.dropna().to_csv(r'../data/generated/topic'+str(i)+'.csv',index=False)'''

#read test data and clean the message column
test = pd.read_csv(r'../data/eval_data.csv')
test_msgs = test.Message.tolist()
stemmed_test_tokens = preprocess_(test_msgs)

temp = [' '.join(i) for i in stemmed_test_tokens]
X_test = pd.Series(temp)

vocablist = [key for key in count_dict if count_dict[key]>5]
hb = HBOS()
topic_outlier=[]
vectorizer = CountVectorizer(vocabulary=vocablist)  

#below loop iterates for every topic, transforms using countvectorizer
#then it fits hbos model on transformed data and predicts decision score of each sentence, 
#finds 75% quantile value which will be our threshold if decision score of a sentence from test set lies below threshold
#for any of the topic then it is a inlier

for i in range(7):
    X = vectorizer.transform(pd.read_csv(r'../data/generated/topic'+str(i)+'.csv')['text'])
    hb.fit(X.toarray())
    
    df = hb.decision_function(X.toarray())
    threshold = np.quantile(df,0.75)
    #plt.hist(df*-1,bins=50)
    temp_ = vectorizer.transform(X_test)
    sdf = hb.decision_function(temp_.toarray())
    topic_outlier.append([0 if i<threshold else 1 for i in sdf ])

    #plt.show()

out_frame = pd.DataFrame(topic_outlier).T


y_test=test['Outlier?'].tolist()
y_pred = []

for row in range(out_frame.shape[0]):
    for i in np.array(out_frame.iloc[row,:]):
        if i==1:
            temp_ = 1
        if i==0:
            temp_ = 0
            break
    #y_pred.append(scipy.stats.mode(np.array(out_frame.iloc[row,:]))[0][0])
    y_pred.append(temp_)

y_pred_ = [True if i==1 else False for i in y_pred]

#model scores
acc_score = metrics.accuracy_score(y_test,y_pred_)
print('\nmodel accuracy score ',acc_score)
prec_score = metrics.precision_score(y_test,y_pred_)
print('\nmodel precision score ',prec_score)