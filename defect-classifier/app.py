import streamlit as st
from PIL import Image
from model_utils import setup_model, predict_image  # Import functions from model_utils.py

# Streamlit UI for image upload
st.title("Manufacturing Defect Classifier")
st.write("Upload an image to classify")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type="jpg")

# Setup the model
model = setup_model(model_path='../final_model.pth')

if uploaded_image is not None:
    # Open the image
    image = Image.open(uploaded_image)

    # Make the prediction
    predicted_class_label, predicted_probability = predict_image(model, image)

    # Display the result
    st.image(uploaded_image, use_container_width=True)
    st.write(f"Prediction: {predicted_class_label}")  # Display the predicted class label
    st.write(f"Probability: {predicted_probability:.2f}%")  # Display the probability
