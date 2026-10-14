import io

from PIL import Image, UnidentifiedImageError
import streamlit as st
from transformers import pipeline

MODEL_NAME = "google/vit-base-patch16-224"
TOP_K = 5


@st.cache_resource(show_spinner=False)
def load_classifier():
    return pipeline("image-classification", model=MODEL_NAME, device=-1)


st.set_page_config(
    page_title="AI image classifier",
    page_icon=":material/image_search:",
    layout="centered",
)

st.title("AI Image Classifier")
st.write("Upload an image to see the model's most likely labels.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        st.error(f"Could not read this image: {error}")
    else:
        image_column, results_column = st.columns(2)
        with image_column:
            st.image(image, alt="Uploaded image for classification", width="stretch")

        with results_column:
            try:
                with st.spinner("Loading the model and classifying the image..."):
                    classifier = load_classifier()
                    predictions = classifier(image, top_k=TOP_K)
            except (OSError, RuntimeError, ValueError) as error:
                st.error(f"Could not classify this image: {error}")
            else:
                st.subheader("Predictions")
                for prediction in predictions:
                    label = prediction["label"]
                    score = prediction["score"]
                    st.progress(score, text=f"{label} — {score:.1%}")
