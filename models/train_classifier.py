import sys
# import libraries
import pandas as pd
import numpy as np
import pickle 

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')# download for lemmatization
nltk.download('punkt')

from sqlalchemy import create_engine

def load_data(database_filepath):
    """Load data from specified database and return lists of X, Y and category names"""
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('MessagesCategories', engine)
    df_Y = df.drop(['genre', 'id', 'message', 'original'], axis=1)
    #convert to lists
    X = df['message'].values
    Y = df_Y.values
    category_names = df_Y.columns
    return X, Y, category_names

def tokenize(text):
    """For a text string, return tokens that are normalized, tokenized to words, without stopwords (english), lemmatized and stemmed"""
    tokens = word_tokenize(text.lower())
    tok = [t for t in tokens if t not in stopwords.words('english')]
    lem = [WordNetLemmatizer().lemmatize(t) for t in tok]
    stem = [PorterStemmer().stem(t) for t in lem]
    return stem


def build_model():
    """Build a pipeline model"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i,name in enumerate(category_names):
        print(name,classification_report(Y_test[:,i], Y_pred[:,i]),'\n')


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()