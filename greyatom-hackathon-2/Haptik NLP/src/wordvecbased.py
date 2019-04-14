from libraries import *

#load train token file
with open(r'../data/train_stem.txt','r',encoding='utf-8') as file_:
    train_stemm = file_.read().splitlines()
train_stem=[]
for i in train_stemm:
    train_stem.append(i.split())

#initialize word2vec
model = Word2Vec(train_stem,size=1000)

#print('length of train',len(train_stem))

#function which return average word2vec vector of each sentence tokens
def get_average_word2vec(tokens_list, vector, generate_missing=False, k=1000):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

#get average word2vec vectors for each sentence, and store it in a list
wv_=[]
for sent in train_stem:
  wv_.append(get_average_word2vec(sent,model))

X= np.array(wv_)
print('shape of training after average word vec',X.shape)

#X = model[model.wv.vocab]
#X= np.array(wv_)

pca = PCA(n_components=2)
result = pca.fit_transform(X)
plt.scatter(result[:, 0], result[:, 1],cmap='viridis')
plt.show()

#as per scatter plot there is no good separation between inliers and outliers, so stopped this model here itself.

'''km = KMeans(n_clusters=3)
km.fit(X)
y_kmeans = km.predict(X)
#plt.scatter(result[:, 0], result[:, 1], c=y_kmeans, s=50, cmap='viridis')
#plt.legend()


#X = model[model.wv.vocab]
X= np.array(wv_)
print(X.shape)
pca = PCA(n_components=2)
result = pca.fit_transform(X)
plt.scatter(result[:, 0], result[:, 1],c=y_kmeans,cmap='viridis')
plt.show()'''