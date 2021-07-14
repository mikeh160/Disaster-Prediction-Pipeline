# Disaster-Response-pipeline-project

### Table of Contents


1. [Project Motivation](#motivation)
2. [File Descriptions](#files)
3. [Instructions](#instructions)
4. [Requirements](#requirements)
5. [Licensing, Authors, and Acknowledgements](#licensing)


## Project Motivation<a name="motivation"></a>

This project aims to build a web-app to classify the messages that have been collated from different sources including Social Media, News and other outlets of information. The data for this project comes from [Appen-FigureEight](https://appen.com/) Inc and it contains the various messages that had been received during natural disasters between 2010 and 2014. 

The web-app that is built in this project would attempt to classify the incoming messages. This would help disaster relief agencies in determining whether the incoming message relates to disaster-relief request and what specific nature of the message received is so that specific help to that requirement is dispatched. We have 36 categories of labels in total and none of the messages are overlapping which would make this problem a multi-classification problem for the machine learning models employed in this project.



## File Descriptions <a name="files"></a>

The description of the files used in this project are:

1. **process_data.py** - this file takes the messages collated from different sources, clean them, aggregates them and saves them in an SQL table to help implement ML models
2. **train_classifier.py** - this file takes the SQL table data and implements machines learning models for classification of the data and saves the result in a pickle file
3. **run.py** - this file contains the Flask back-end for the web-app that would take user input and attempt to classify the message recieved
4. **ETL pipeline preparation.py** - this file shows the development phase of process_data.py
5. **ML pipeline preparation.py** - this file shows the development stages for the train_classifier.py


## Instructions<a name="instructions"></a>

1. Run the following commands in the project's root directory to run this project

   * To run the ETL pipeline that cleans data and prepares it in the required SQL format : python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db
   * To run the ML pipeline that would classify the labels using Machine Learning models and saves the results in pickle file : python models/train_classifier.py data/disaster_response.db models/classifier.pkl

2. Run the following command in the app's directory to run your web app: `python run.py`

3. Go to http://0.0.0.0:3001/ to see the results of the project and type in some messages to see the relationships

4.The graphical output of the project on the webpage is represented in a screenshot below 

<p align="center">
  <img width="460" height="460" src="https://user-images.githubusercontent.com/27803552/125647224-a9d5a0ff-4556-41ab-b2c8-f411607e410e.png">
</p>


## Requirements<a name ="requirements"></a>

Following libraries would be required to run the project

* NLTK for natural language processing
* Pandas, Numpy, Scikit-Learn for data manipulation and machine learning
* Plotly for data visualization
* Flask for back-end web-app

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Incredibly grateful to the team at Udacity as well as FigureEight for providing Data required for the completition of this project
