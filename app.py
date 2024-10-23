import streamlit as st
# from transformers import ViTForImageClassification, ViTFeatureExtractor
# import torch
# from torchvision import transforms
# from PIL import Image
# import numpy as np

# Load pre-trained model and feature extractor
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Define a function for image prediction
def predict(image):
    # Define image transformations
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the expected input size
        transforms.ToTensor(),           # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    # Preprocess the image
    image = image_transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        logits = model(image).logits
        predicted_class = logits.argmax(-1).item()  # Get the predicted class index
        confidence = torch.nn.functional.softmax(logits, dim=1)[0][predicted_class].item()  # Get confidence score

    return predicted_class, confidence

# Define a mapping for class indices to disease names (update this based on your dataset)
class_names = {
    0: "Healthy",
    1: "Powdery Mildew",
    2: "Downy Mildew",
    3: "Leaf Blight",
    4: "Fusarium Wilt",
    5: "Bacterial Blight",
    6: "Root Rot",
    7: "Anthracnose",
    8: "Rust",
    9: "Cercospora Leaf Spot",
    10: "Gray Mold",
    11: "Tomato Blight",
    12: "Black Spot",
    13: "Crown Gall",
    14: "Citrus Canker",
    15: "Late Blight",
    16: "White Rust",
    17: "Mosaic Virus",
    18: "Sclerotinia Rot",
    19: "Phytophthora Root Rot",
}


# Streamlit app layout
st.title("Plant Disease Detection")
st.write("Upload an image of a diseased plant to predict the disease type.")

# Upload image
uploaded_file = st.file_uploader("Choose a plant image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prediction
    st.write("Classifying...")
    predicted_class, confidence = predict(image)

    # Display results
    st.write(f"Predicted Disease: {class_names.get(predicted_class, 'Unknown')} (Confidence: {confidence:.2f})")

# Run the app with the command: streamlit run your_script_name.py
