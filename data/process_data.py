import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import sqlite3

def load_data(messages_filepath, categories_filepath):

    '''
    Reads the data from the given filepaths and merges them into a single dataframe(df)
    
        inputs:
            csv file paths
        returns:
            dataframe(df)
    '''
    
    messages = pd.read_csv(str(messages_filepath))
    categories = pd.read_csv(str(categories_filepath))
    
    df = messages.merge(categories, how = 'left', on = 'id')
    
    return df
    #pass


def clean_data(df):

    ''' Takes the raw data df, manipulates the data shape into desired SQL table, cleans and saves the result  
    
        *inputs:
            df
        *returns:
            df
    '''
    
    categories = df.categories.str.split(pat=';', n=-1, expand = True)
    row = categories.loc[0,:]
    category_colnames = row.map(lambda v: v.split('-')[0])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.extract('.*(\d{1})', expand = False)
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    df.drop('categories', 1, inplace = True)
    df = pd.concat([df, categories], join = 'inner', sort = False, axis = 1)
    df.groupby(df.columns.tolist(), as_index = False).size().reset_index().rename(columns = {0:"duplicates"})
    print('No. of rows vs columns are:', df.shape)
    df.drop_duplicates(inplace = True)
    df.groupby(df.columns.tolist(), as_index = False).size().reset_index().rename(columns = {0:"duplicates"})
    #Removing rows containing the value 2 in related column, since we 're only working with binary assumptions 
    #here with 0 meaning message is not related and a 1 pointing to a message relating to specific category
    df = df[df.related != 2]
    #pass
    return df

def save_data(df, database_filename):

    '''
    Takes a dataframe(df) and filepath and saves the result of that dataframe(df) into an sql table
    
        *inputs:
            df, df_filepath
        *returns:
            saves an SQL table
    '''

    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_response_local_name', engine, if_exists = 'replace' , index=False)
    engine.dispose()
    #pass  


def main():
    
    '''
    Specifying the program input in message_filepath, categories_filepath & db_filepath to run the main program
    
        *inputs:
            user sppecified input specifying messages_filepath, categories_filepath and database_filepath in terminal
        *returns:
            runs the entire process_data.py modules
    '''
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
        
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()