# Importing libraries
from flask import Flask,request,jsonify
import pickle
import numpy as np
import re
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer

# Creating an instance of Flask application
app = Flask(__name__)

# Opening the model and vectorizer saved from Jupyter Notebook
model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

# Setting up spacy and sentiment intensity analyzer model
nlp = spacy.load('en_core_web_sm')
vader = SentimentIntensityAnalyzer()

# Setting up URL of the application
@app.route('/api',methods=['POST'])
def prediction_api():
    
    # Retrieving data from user's JSON file
    data = request.json['tweets']

    # Checking for any negation terms
    if re.search(r"\b(?:not|n't|no|never|none|nothing|nowhere|neither|nor|nobody)\b",str(data)):

        # Computing the sentiment intensity of the tweet
        vader_compute = vader.polarity_scores(str(data))['compound']

        # Label is good if score is greater than or equal to 0.2
        if vader_compute >= 0.2:
            prediction = 'good'
        
        # Label is bad if score is less than or equal to -0.2
        elif vader_compute <= -0.2:
            prediction = 'bad'
        
        # Label is neutral if score is between 0.2 and -0.2
        else:
            prediction = 'neutral'
        
        # Return prediction
        return jsonify(prediction)
    else:
        
        # Removing if there are any URLs
        try:
            data = re.sub(r"https[s]?://\S+", '', data)
        except Exception as e:
            print("Error in URL removal:", e)
        
        # Removing if there are any punctuations
        try:
            data = re.sub(r"[^\w\s]", '', data)
        except Exception as e:
            print("Error in non-alphanumeric removal:", e)
        
        # Removing if there are any alpahanumeric values
        try:
            data = re.sub(r"\d+", '', data)
        except Exception as e:
            print("Error in digit removal:", e)
        
        # Applying lemmatization
        data = nlp(str(data))
        data = [x.lemma_ for x in data]
        data = ' '.join(data)
        data = [data]

        # Applying TFID-vectorization
        data = vectorizer.transform(data)

        # Making predictions
        prediction = model.predict(data)

        # Return prediction
        return jsonify(prediction[0])

# Run the application
if __name__ == '__main__':
    app.run(debug=True)