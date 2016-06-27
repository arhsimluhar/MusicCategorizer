from __future__ import division
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import unicodedata
import codecs
import string
import pickle
stemmer = SnowballStemmer('english')
cachedStopWords = stopwords.words("english")
data=[]
with codecs.open('finalLyricsList2.txt','r') as f:
	lines=f.readlines()
	count=0
	for line in lines:

		labels,features=line.split('**/**')		
		features = unicode(features, "utf-8")
		features = unicodedata.normalize('NFKD',features).encode('ascii','ignore')
		features=features.translate(None, string.punctuation)
		features=features.lower()  
		features=[word for word in features.split() if word not in cachedStopWords]
		for word in features:
			stemmer.stem(word)
		features=" ".join(features)
		data.append([labels,features])
		count+=1
		if(count%500==0):
			print "completed:",count/len(lines)*100
		

	genres={} 
	for line in lines:
		labels,features=line.split('**/**')		
		try:
			genres[labels]+=1
		except:
			genres[labels]=0
	print "File composition"
	for g in genres.keys():
		print g+": ",genres[g]
	
#saving into serialized object
stem_data=open("stemmed_text",'wb')
pickle.dump(data,stem_data)
stem_data.close()



from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
email=[]
bow=vectorizer.fit(email)
bow=vectorizer.transform(email)
vectorizer.vocabulary_.get("great")

#stemmer

 