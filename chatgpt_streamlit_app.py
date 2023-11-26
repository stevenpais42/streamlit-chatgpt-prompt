import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Display the title
st.title("Simple Iris Flower Prediction App")

# Display a description
st.write("This app predicts the Iris flower type based on user input parameters.")

# Add a sidebar with sliders for user input parameters
st.sidebar.header("User Input Parameters")

sepal_length = st.sidebar.slider("Sepal Length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.sidebar.slider("Sepal Width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.sidebar.slider("Petal Length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.sidebar.slider("Petal Width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Display user input below the sidebar
st.sidebar.write("User Input:")
st.sidebar.write(f"Sepal Length: {sepal_length}")
st.sidebar.write(f"Sepal Width: {sepal_width}")
st.sidebar.write(f"Petal Length: {petal_length}")
st.sidebar.write(f"Petal Width: {petal_width}")

# Make predictions using user-input parameters
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)

# Display the predicted class label
st.subheader("Prediction:")
st.write(f"The predicted class label is: {iris.target_names[prediction]}")

# Display class labels and their corresponding index numbers
st.subheader("Class Labels and Index Numbers:")
for i, class_label in enumerate(iris.target_names):
    st.write(f"{class_label}: {i}")

# Show the prediction probability for each class label
st.subheader("Prediction Probabilities:")
proba_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
st.write(proba_df)
