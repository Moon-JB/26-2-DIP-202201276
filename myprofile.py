import streamlit as st
from PIL import Image

st.title("Object counting")

# Sidebar
st.sidebar.header("Image Upload")

uploaded_file = st.sidebar.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# Display uploaded image
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Input Image")
    st.image(image, caption="Uploaded Image")
