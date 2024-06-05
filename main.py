import streamlit as st
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

# Initialize OCR model
@st.cache(allow_output_mutation=True)
def initialize_ocr(lang):
    return PaddleOCR(use_angle_cls=True, lang=lang)

# Define function to perform OCR on uploaded image
def perform_ocr(image, ocr_model):
    image_np = np.array(image)
    result = ocr_model.ocr(image_np, cls=True)
    return result

# Streamlit UI
st.title("English OCR App")
st.write("Upload an image for text extraction:")

# Allow user to upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Initialize OCR model with English language
    ocr_model = initialize_ocr("en")

    # Load image
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform OCR
    result = perform_ocr(image, ocr_model)

    # Display OCR results
    recognized_text = ""
    for res in result:
        for line in res:
            st.write(f"Text: {line[1][0]}, Confidence: {line[1][1]:.2f}")
            recognized_text += line[1][0] + "\n"

    # Draw OCR result on image
    boxes = [line[0] for line in result[0]]
    txts = [line[1][0] for line in result[0]]
    scores = [line[1][1] for line in result[0]]
    im_show = draw_ocr(np.array(image), boxes, txts, scores, font_path="font-times-new-roman.ttf")
    im_show = Image.fromarray(im_show)
    st.image(im_show, caption="Result", use_column_width=True)

    # Display recognized text in a larger text area
    text_area = st.text_area("Recognized Text", recognized_text, height=300)

    st.write("Copy the text manually from the text area above.")
