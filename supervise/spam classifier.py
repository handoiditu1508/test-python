import os
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#clear console
clear = lambda: os.system('cls')
clear()

messages = ["free prizes for you",
	"hey, can I borrow your prize",
	"you have to submit that test",
	"this is a free message, free for you",
	"free free free, that right free shit"
]
classes = ["spam", "ham", "ham", "spam", "spam"]

#count words in message array
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(messages)

print(vectorizer.get_feature_names())
print()
#['borrow', 'can', 'for', 'free', 'have', 'hey', 'is', 'message', 'prize',
# 'prizes', 'right', 'shit', 'submit', 'test', 'that', 'this', 'to', 'you', 'your']

print(counts.toarray())
print()
#[[0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0]
# [1 1 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 1]
# [0 0 0 0 1 0 0 0 0 0 0 0 1 1 1 0 1 1 0]
# [0 0 1 2 0 0 1 1 0 0 0 0 0 0 0 1 0 1 0]
# [0 0 0 4 0 0 0 0 0 0 1 1 0 0 1 0 0 0 0]]

#create classifier using Naive Bayes theorem on data(counts) and result(classes)
classifier = MultinomialNB()
classifier.fit(counts, classes)

#try predicting test messages to be "spam" or "ham"
testMessages = ["Free money! now!", "I need to borrow your money"]
testCounts = vectorizer.transform(testMessages)#converts [testCounts] to the same model as [counts]
predictions = classifier.predict(testCounts)
print(predictions)