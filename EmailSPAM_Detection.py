#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# In[ ]:


nltk.download("stopwords")


# In[ ]:


data = pd.read_csv("emails.csv", encoding="latin-1")
data = data[["v1", "v2"]]  # Select relevant columns


# In[ ]:


data.columns = ["label", "text"]
data["label"] = data["label"].map({"ham": 0, "spam": 1})  # Map labels to 0 (ham) and 1 (spam)


# In[ ]:


stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


# In[ ]:


def preprocess_text(text):
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    text = " ".join([stemmer.stem(word) for word in text.split()])
    return text


# In[ ]:


data["text"] = data["text"].apply(preprocess_text)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2, random_state=42)


# In[ ]:


# Create a pipeline with CountVectorizer and MultinomialNB
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB())
])


# In[ ]:


# Define hyperparameters for tuning
param_grid = {
    'vect__ngram_range': [(1, 1), (1, 2)],  # Unigrams or bigrams
    'clf__alpha': [0.1, 0.5, 1.0],  # Smoothing parameter
}


# In[ ]:


# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(text_clf, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)


# In[ ]:


best_clf = grid_search.best_estimator_


# In[ ]:


y_pred = best_clf.predict(X_test)


# In[ ]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# In[ ]:


print("Best Parameters:", grid_search.best_params_)
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(report)
print(f"ROC AUC: {roc_auc:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


# In[ ]:


# Function to predict whether a custom message is spam or not
def predict_custom_message(message):
    message = preprocess_text(message)
    prediction = best_clf.predict([message])[0]
    if prediction == 0:
        return "Not Spam"
    else:
        return "Spam"


# In[ ]:


# Example usage of the custom message prediction function
custom_message = "Congratulations! You've won a free gift. Claim it now!"
result = predict_custom_message(custom_message)
print(f"Custom Message: '{custom_message}' is {result}")


# In[ ]:




