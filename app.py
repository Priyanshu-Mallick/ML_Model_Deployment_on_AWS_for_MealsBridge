from flask import Flask, request, jsonify
import joblib
import boto3

app = Flask(__name__)

# Load model from S3
s3 = boto3.client('s3')
s3.download_file('mealsbridge-data-storage-12345', 'models/food_wastage_model.joblib', 'food_wastage_model.joblib')
model = joblib.load('food_wastage_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
