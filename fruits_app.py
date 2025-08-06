import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('fruits_model.h5')

model = load_model()

st.title("🧠 Fruit classification App")
st.write("Upload an image of 1-apple 2-banana 3 orange.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image (example: 224x224 input for model)
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension

    # Predict
    prediction = model.predict(img_array)
    
    # Assuming classification with softmax

    predicted_class = np.argmax(prediction, axis=1)[0]

    if predicted_class == 0:
        st.success("Predicted class: Apple")
    elif predicted_class == 1:
        st.success("Predicted class: Banana")
    elif predicted_class == 2:
        st.success("Predicted class: Orange")
    else:
        st.error("it is not a fruit")

