import streamlit as st
import torch
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import tempfile
import os
import torch.nn as nn
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Sketch to Photo - CycleGAN", page_icon="🎨", layout="wide")

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_res_blocks=6):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        in_f, out_f = 64, 128
        for _ in range(2):
            model += [nn.Conv2d(in_f, out_f, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_f), nn.ReLU(inplace=True)]
            in_f, out_f = out_f, out_f * 2
        for _ in range(n_res_blocks):
            model += [ResidualBlock(in_f)]
        out_f = in_f // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_f, out_f, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_f), nn.ReLU(inplace=True)]
            in_f, out_f = out_f, out_f // 2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, out_channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model():
    MODEL_URL = "https://github.com/Mustehsan-Nisar-Rao/Cyclic-GAN/releases/download/v.1/cyclegan_best_model.pth"
    weights_path = "/tmp/cyclegan_weights.pth"

    if not os.path.exists(weights_path):
        with st.spinner("Model download ho raha hai..."):
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            total = int(response.headers.get('content-length', 0))
            progress = st.progress(0)
            downloaded = 0
            with open(weights_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            progress.progress(min(downloaded / total, 1.0))
            progress.empty()

    try:
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    except Exception as e:
        os.remove(weights_path)
        st.error(f"Model load error: {e}")
        return None

    model = Generator()
    model.load_state_dict(checkpoint['G_S2P'])
    model.eval()
    return model

def preprocess(image):
    image = image.resize((128, 128))
    img = np.array(image).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

def postprocess(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0).numpy()
    tensor = (tensor + 1.0) / 2.0
    tensor = np.clip(tensor, 0, 1)
    return Image.fromarray((tensor * 255).astype(np.uint8))

def generate(img, model):
    with torch.no_grad():
        tensor = preprocess(img)
        output = model(tensor)
    return postprocess(output)

st.title("Sketch to Photo Translation")
st.markdown("*Powered by CycleGAN*")

model = load_model()
if model is None:
    st.error("Model load nahi hua!")
    st.stop()
st.success("Model ready!")

tab1, tab2 = st.tabs(["Draw Sketch", "Upload Sketch"])

with tab1:
    st.markdown("Neeche sketch banao aur Generate button dabao!")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Drawing Canvas")
        brush_size  = st.slider("Brush size", 1, 30, 8)
        brush_color = st.color_picker("Brush color", "#000000")

        canvas_result = st_canvas(
            fill_color       = "rgba(255, 255, 255, 0)",
            stroke_width     = brush_size,
            stroke_color     = brush_color,
            background_color = "#FFFFFF",
            height           = 400,
            width            = 400,
            drawing_mode     = "freedraw",
            key              = "canvas",
        )

        generate_btn = st.button("Generate Photo", type="primary", use_container_width=True)

    with col2:
        st.subheader("Generated Photo")
        if generate_btn:
            if canvas_result.image_data is not None:
                img_array  = canvas_result.image_data.astype(np.uint8)
                canvas_img = Image.fromarray(img_array).convert("RGB")
                img_np     = np.array(canvas_img)

                if img_np.mean() > 250:
                    st.warning("Pehle kuch draw karo!")
                else:
                    with st.spinner("Generating..."):
                        result = generate(canvas_img, model)
                    st.image(result, use_container_width=True, caption="Generated Photo")
                    buf = BytesIO()
                    result.save(buf, format="PNG")
                    st.download_button(
                        "Download Photo",
                        buf.getvalue(),
                        "generated_photo.png",
                        mime="image/png",
                        use_container_width=True
                    )
            else:
                st.warning("Pehle kuch draw karo!")
        else:
            st.info("Left side pe sketch banao phir Generate dabao!")

with tab2:
    st.markdown("Sketch upload karo!")
    uploaded = st.file_uploader("Image choose karo", type=['png', 'jpg', 'jpeg'])

    if uploaded:
        sketch = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input Sketch")
            st.image(sketch, use_container_width=True)
        with col2:
            st.subheader("Generated Photo")
            with st.spinner("Generating..."):
                result = generate(sketch, model)
            st.image(result, use_container_width=True)
            buf = BytesIO()
            result.save(buf, format="PNG")
            st.download_button(
                "Download Photo",
                buf.getvalue(),
                "generated_photo.png",
                mime="image/png",
                use_container_width=True
            )

st.sidebar.markdown("""
**Model:** CycleGAN  
**Training:** 50 epochs  
**Dataset:** Sketchy Database  
**Image size:** 128x128

**How to use:**  
1. Draw tab mein sketch banao  
2. Ya Upload tab mein image upload karo  
3. Generate button dabao  
4. Photo download karo!
""")
