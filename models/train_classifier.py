import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
import numpy as np
import pandas as pd
from sqlalchemy import create_engine 
import sys

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline

import pickle 

def load_data(database_filepath):

'''
Takes the SQL table and converts the result into a df. Furthermore, breaksdown the data into input data(X) and target data(y)

    *inputs : 
            database_filepath
    *returns: 
            Input_data(X), target_data(y) and column_names of df(colnames)
'''
    
    #engine = create_engine('sqlite:///../data/disaster_response.db')
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    conn = engine.connect()
    df = pd.read_sql_table('disaster_response_local_name', conn)
    X = df.message.values
    y = df.iloc[:, 4:]
    colnames = y.columns
    engine.dispose()
    #pass
    return X, y, colnames

def tokenize(text):

'''
Takes the text data, tokenizes the data into words, removes non-essential words not serving any purpose, evaluates the realtionship to its context and returns the clean tokens

    *inputs : 
            text
    *returns: 
            clean tokens
'''
    
    tokens = word_tokenize(text)
    tokens_wihtout_sw = [w for w in tokens if w not in stopwords.words("english") ]
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []    
    for tok in tokens_wihtout_sw:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens
    #pass


def build_model():

'''
A pipeline is defined to run consecutive operations in succession inlcuding the multiple classification algorithm, while parameters for choosing ideal model are searched using grid search. The results of the model are saved

    *inputs : 
            None
    *returns: 
            Model results (model_pipeline_cv)
'''
    
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)), 
            ('tfidf', TfidfTransformer())
        ])),
        ('mclf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        #'mclf__estimator__min_samples_split': [2, 3],
        'mclf__estimator__n_estimators': [50, 70],         
    }

    model_pipeline_cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs = 4, cv = 2, verbose = 3)
    #print(cv.best_params_)
    return model_pipeline_cv
    #pass


def evaluate_model(model, X_test, Y_test, category_names):
'''
Takes the results of the model and tries to predict the target classification using machine learning algorithms on the test_dataset(X_test). Displays the final results into a confusion matrix

    *inputs : 
            saved model configuration(model), test input(X_test), training output(Y_test), and category_names of columns
    *returns: 
            displays results of confusion matrix
'''
    
    y_pred = model.predict(X_test)
    target_dataframe = pd.DataFrame(y_pred, columns = category_names)
    
    for i, value in enumerate(target_dataframe):
        print("Model Confusion Matrix for the specific keyword :", value, "are below")
        print(classification_report(Y_test.iloc[:,i], target_dataframe.iloc[:,i] ))
    #pass


def save_model(model, model_filepath):
'''
This module saves the configuration of the model so that long running processes of model prediction (in our case almost more than half an hour) do not have to be run again and again

    *inputs : 
            model, model_filepath
    *returns: 
            saves the results in a pickle file (classifier.pkl)
'''
    
    pickle.dump(model, open(model_filepath, 'wb') )
    #pass


def main():
'''
Takes the user specified input of database filepath, model filepath and, splits the dataset into training and test dataset and runs all the modules of the main program including building model, training model, evaluating model and then finally saving model

    *inputs : 
            user specified database_filepath, model_filepath
    *returns: 
            saves model configuration into a pickle file
'''
    
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