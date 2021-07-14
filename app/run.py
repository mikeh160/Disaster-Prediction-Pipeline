import json
import plotly
import pandas as pd

import plotly.graph_objects as go

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

import pickle


app = Flask(__name__)


def tokenize(text):

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///..//data/disaster_response.db')
df = pd.read_sql_table('disaster_response_local_name', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #extracting and manipulating data now to find the five categories with greater received data
    #for this we'd would need to transpose the dataset to find categories i.e. keywords received in descending order
    
    t_df = df.iloc[:,4:].apply(pd.Series.value_counts)
    t_df = t_df.transpose()
    t_df = t_df.reset_index().rename(columns={'index': 'types_of_msgs'})
    #dropping the reduntant column 2 which has an unidentified value 2 for only one keyword : "related"
    t_df = t_df.drop([2], axis = 1) 
    #arranging in descending order
    sorted_df = t_df.sort_values(by = [1], ascending=False)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [ {
                #Bar(
                #    x=genre_names,
                #    y=genre_counts
                #)
                    'values' : genre_counts,
                    'labels' : genre_names,   
                    'type' : 'pie', 
                    'hole' : 0.4,
                'marker': {
                'colors': 'ultimateColors[3]'
                }
                #)
            }
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count" },
                'xaxis': {
                    'title': "Genre"} }
        }, 
        
    { 'data': [ {
              #  Bar(
                    'x':sorted_df.iloc[:5,0],
                    'y':sorted_df.iloc[:5,2],
                    'type' :'bar', 
            'marker': {
                    'color': 'indianred'
                    }
              #  )    
                }],
        
        'layout': {
                'title': 'Distribution of 5 Most Important Keywords Received',
                'yaxis': {
                    'title': "Count" },
                'xaxis': {
                    'title': "Keyword"} }
        } 
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():    
    
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()