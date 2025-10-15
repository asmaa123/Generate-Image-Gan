
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🎨 AI Image Generator")
st.markdown("Enter a prompt in English and customize the generation parameters.")

# Advanced options sidebar
st.sidebar.header("Advanced Options")
width = st.sidebar.slider("Image Width", 256, 1024, 512, step=64)
height = st.sidebar.slider("Image Height", 256, 1024, 512, step=64)
num_inference_steps = st.sidebar.slider("Inference Steps", 10, 100, 50, step=5)
guidance_scale = st.sidebar.slider("Guidance Scale", 1.0, 20.0, 7.5, step=0.5)

# Prompt input
prompt = st.text_input("Enter your prompt (English only):")

# Load model with caching
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "sd-legacy/stable-diffusion-v1-5",
        torch_dtype=torch.float32  # float16 لا تعمل على CPU
    )
    pipe = pipe.to(device)
    return pipe

pipe = load_model()

# Generate button
if st.button("Generate"):
    if prompt.strip() == "":
        st.warning("⚠️ Please enter a prompt in English!")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Callback function لتحديث البار لكل خطوة
        def callback(step: int, timestep: int, latents):
            progress = int((step + 1) / num_inference_steps * 100)
            if progress > 100:
                progress = 100
            progress_bar.progress(progress)
            progress_bar.progress(progress)
            status_text.text(f"Generating image... {progress}%")

        with st.spinner("Generating image..."):
            image = pipe(
                prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                callback=callback,
                callback_steps=1
            ).images[0]

        # عرض الصورة
        st.image(image, caption="Generated Image", use_column_width=True)

        # زر تحميل
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        st.download_button(
            label="Download Image",
            data=buf,
            file_name="generated_image.png",
            mime="image/png"
        )
        st.success("✅ Done!")
