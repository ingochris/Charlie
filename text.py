categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

import sklearn.datasets.twenty_newsgroups as news
from sklearn.feature_extraction.text import CountVectorizer

twenty_train = news.fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print X_train_counts.shape

print "hello!"