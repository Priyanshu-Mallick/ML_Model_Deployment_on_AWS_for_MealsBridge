import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data
data = pd.read_csv('s3://mealsbridge-data-storage-12345/food_wastage_data.csv')

# Preprocess data
X = data.drop('wastage', axis=1)
y = data['wastage']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Model accuracy: {accuracy}')

# Save model
joblib.dump(model, 'food_wastage_model.joblib')

# Upload model to S3
s3 = boto3.client('s3')
s3.upload_file('food_wastage_model.joblib', 'mealsbridge-data-storage-12345', 'models/food_wastage_model.joblib')
