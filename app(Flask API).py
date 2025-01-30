from flask import Flask, request, jsonify, render_template
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = pk.load(open("model.pkl", "rb"))

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2500)

@app.route('/')
def home():
    return render_template('index.html')  # HTML template for user input

@app.route('/predict', methods=['POST'])
def predict():
    # Get the review from the form
    review = request.form['review']
    
    # Transform the review into a vector
    transformed_review = vectorizer.fit_transform([review]).toarray()
    
    # Predict using the loaded model
    prediction = model.predict(transformed_review)
    
    # Map prediction to label (if required, e.g., positive/negative)
    if prediction[0] == 1:
        result = "Positive Review"
    else:
        result = "Negative Review"
    
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
