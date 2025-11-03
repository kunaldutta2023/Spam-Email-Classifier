from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    data = [message]
    transformed = vectorizer.transform(data)
    prediction = model.predict(transformed)[0]

    result = "🚨 Spam Message" if prediction == 1 else "✅ Not Spam"
    return render_template('index.html', prediction_text=result, input_text=message)

if __name__ == "__main__":
    app.run(debug=True)
