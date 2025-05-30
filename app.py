import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognizer", layout="centered")

st.title("Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) in the box below and the CNN will predict what it is.")

@st.cache(allow_output_mutation=True)
def load_trained_model():
    model = load_model('my_cnn_model.h5')
    return model

model = load_trained_model()

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    img = cv2.resize(img, (28, 28))
    img = 255 - img
    img = img / 255.0
    img_input = np.expand_dims(img, axis=0)[..., np.newaxis]

    st.image(img, caption="Processed Image", width=140)
    
    prediction = model.predict(img_input)
    predicted_digit = np.argmax(prediction)
    st.write("**Predicted Digit:**", predicted_digit)
