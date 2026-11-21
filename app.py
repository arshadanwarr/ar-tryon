import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

import tryon

st.set_page_config(page_title="AR Try-On", page_icon="🕶️", layout="wide")


@st.cache_resource
def load_resources():
    return tryon.create_landmarker(), tryon.load_assets()


landmarker, assets = load_resources()

st.title("AR Try-On")
st.write("Take a photo or upload one, choose accessories, and they are placed on your face "
         "using facial landmarks detected by MediaPipe.")

with st.sidebar:
    st.header("Accessories")
    use_glasses = st.checkbox("Glasses", True)
    use_hat = st.checkbox("Hat", True)
    use_mustache = st.checkbox("Mustache", False)
    st.header("Fit")
    glasses_scale = st.slider("Glasses size", 1.6, 3.0, 2.2, 0.05)
    hat_scale = st.slider("Hat size", 1.0, 2.2, 1.5, 0.05)
    mustache_scale = st.slider("Mustache size", 0.8, 2.0, 1.4, 0.05)
    headroom = st.checkbox("Add space above the head (so hats are not cut off)", True)

source = st.radio("Photo source", ["Camera", "Upload"], horizontal=True)
if source == "Camera":
    file = st.camera_input("Take a photo")
else:
    file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

items = [name for name, on in [("glasses", use_glasses), ("hat", use_hat),
                               ("mustache", use_mustache)] if on]

if file is not None:
    image = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    scale = 1280 / max(frame.shape[:2])          # keep large photos fast
    if scale < 1:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    if not items:
        st.info("Select at least one accessory in the sidebar.")
    else:
        if headroom and "hat" in items:
            frame = tryon.add_headroom(frame)
        result, found = tryon.apply_tryon(frame, landmarker, assets, items,
                                          glasses_scale, hat_scale, mustache_scale)
        if not found:
            st.warning("No face detected. Try a clear, front-facing photo with good lighting.")
        else:
            col1, col2 = st.columns(2)
            col1.image(image, caption="Original", use_container_width=True)
            col2.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                       caption="With try-on", use_container_width=True)
            ok, buf = cv2.imencode(".png", result)
            st.download_button("Download result", buf.tobytes(),
                               file_name="tryon.png", mime="image/png")