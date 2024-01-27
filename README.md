# house_price_prediction2
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load your dataset
# For this example, let's assume you have a CSV file named 'house_data.csv'
# with features (e.g., bedrooms, bathrooms, square footage, etc.) and target (price).
data = pd.read_csv('house_data.csv')

# Extract features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression Model (using scikit-learn)
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
linear_predictions = linear_model.predict(X_test_scaled)

# Evaluate the Linear Regression Model
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_predictions))
print(f'Linear Regression RMSE: {linear_rmse}')

# Neural Network Model (using TensorFlow)
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)

# Predictions using the Neural Network Model
nn_predictions = model.predict(X_test_scaled).flatten()

# Evaluate the Neural Network Model
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_predictions))
print(f'Neural Network RMSE: {nn_rmse}')
