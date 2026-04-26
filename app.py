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
import time

st.set_page_config(page_title="Sketch to Photo - CycleGAN", page_icon="🎨", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
.stButton > button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
}
.stAlert {
    padding: 1rem;
    border-radius: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

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
    MODEL_URL = "https://github.com/Abdulbaset1/Domain-Adaptation-and-Unpaired-Image-to--Image-Translation-using-CycleGAN/releases/download/v1/cyclegan_model.pth"
    weights_path = "/tmp/cyclegan_weights.pth"

    if not os.path.exists(weights_path):
        with st.spinner("📥 Downloading model (this may take a few minutes)..."):
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                total = int(response.headers.get('content-length', 0))
                progress_bar = st.progress(0)
                status_text = st.empty()
                downloaded = 0
                
                with open(weights_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                progress = min(downloaded / total, 1.0)
                                progress_bar.progress(progress)
                                status_text.text(f"Downloaded: {downloaded/1024/1024:.1f} MB / {total/1024/1024:.1f} MB")
                
                progress_bar.empty()
                status_text.empty()
                st.success("✅ Model downloaded successfully!")
                
            except Exception as e:
                st.error(f"❌ Download failed: {str(e)}")
                return None

    try:
        # Try to load with weights_only=False first
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    except Exception as e:
        st.warning(f"First attempt failed: {e}")
        try:
            # Fallback to regular load
            checkpoint = torch.load(weights_path, map_location='cpu')
        except Exception as e2:
            st.error(f"❌ Model load error: {e2}")
            if os.path.exists(weights_path):
                os.remove(weights_path)
            return None

    # Check if model has the expected keys
    if 'G_S2P' not in checkpoint:
        st.error("❌ Invalid model file: 'G_S2P' key not found")
        return None
        
    model = Generator()
    try:
        model.load_state_dict(checkpoint['G_S2P'])
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model state: {e}")
        return None

def preprocess(image):
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize((128, 128), Image.Resampling.LANCZOS)
    img = np.array(image).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

def postprocess(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    tensor = (tensor + 1.0) / 2.0
    tensor = np.clip(tensor, 0, 1)
    return Image.fromarray((tensor * 255).astype(np.uint8))

def generate(img, model):
    start_time = time.time()
    with torch.no_grad():
        tensor = preprocess(img)
        output = model(tensor)
    result = postprocess(output)
    st.info(f"✨ Generation completed in {time.time() - start_time:.2f} seconds")
    return result

# Main UI
st.title("🎨 Sketch to Photo Translation")
st.markdown("*Powered by CycleGAN - Transform your sketches into realistic photos!*")

# Load model
with st.spinner("🔄 Loading AI model..."):
    model = load_model()

if model is None:
    st.error("❌ Model failed to load. Please check your internet connection and try again.")
    st.stop()

st.success("✅ Model ready! Start creating!")

# Create tabs
tab1, tab2, tab3 = st.tabs(["✏️ Draw Sketch", "📤 Upload Sketch", "ℹ️ About"])

with tab1:
    st.markdown("### Create your masterpiece!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎨 Drawing Canvas")
        
        # Drawing controls
        brush_size = st.slider("Brush size", 1, 30, 8, key="brush_size")
        brush_color = st.color_picker("Brush color", "#000000", key="brush_color")
        
        # Canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=brush_size,
            stroke_color=brush_color,
            background_color="#FFFFFF",
            height=400,
            width=400,
            drawing_mode="freedraw",
            key="canvas_draw",
        )
        
        generate_btn = st.button("🚀 Generate Photo", type="primary", use_container_width=True)
        
    with col2:
        st.subheader("📸 Generated Photo")
        
        if generate_btn:
            if canvas_result.image_data is not None:
                img_array = canvas_result.image_data.astype(np.uint8)
                canvas_img = Image.fromarray(img_array).convert("RGB")
                img_np = np.array(canvas_img)
                
                # Check if canvas is empty (mostly white)
                if img_np.mean() > 250:
                    st.warning("⚠️ Please draw something on the canvas first!")
                else:
                    with st.spinner("🎨 AI is transforming your sketch..."):
                        result = generate(canvas_img, model)
                    
                    st.image(result, use_container_width=True, caption="✨ AI Generated Photo")
                    
                    # Download button
                    buf = BytesIO()
                    result.save(buf, format="PNG")
                    st.download_button(
                        "💾 Download Photo",
                        buf.getvalue(),
                        "generated_photo.png",
                        mime="image/png",
                        use_container_width=True
                    )
            else:
                st.warning("⚠️ Please draw something on the canvas first!")
        else:
            st.info("👈 Draw a sketch on the left, then click Generate!")

with tab2:
    st.markdown("### Upload your sketch")
    
    uploaded = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload a clear sketch for best results"
    )
    
    if uploaded:
        sketch = Image.open(uploaded).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Input Sketch")
            st.image(sketch, use_container_width=True)
            
            # Show image info
            st.caption(f"Size: {sketch.size[0]}x{sketch.size[1]} | Mode: {sketch.mode}")
            
        with col2:
            st.subheader("✨ Generated Photo")
            
            if st.button("🎨 Generate from Upload", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing your sketch..."):
                    result = generate(sketch, model)
                
                st.image(result, use_container_width=True, caption="✨ AI Generated Photo")
                
                buf = BytesIO()
                result.save(buf, format="PNG")
                st.download_button(
                    "💾 Download Photo",
                    buf.getvalue(),
                    "generated_photo_from_upload.png",
                    mime="image/png",
                    use_container_width=True
                )

with tab3:
    st.markdown("""
    ## About This App
    
    ### 🎯 What is CycleGAN?
    CycleGAN is a powerful deep learning model that can translate images from one domain to another without paired examples. This app uses CycleGAN to transform sketches into realistic photos!
    
    ### 🚀 Features
    - ✏️ **Draw sketches** directly in the browser
    - 📤 **Upload existing** sketches
    - 🎨 **AI-powered** transformation
    - 💾 **Download** generated photos
    - ⚡ **Fast processing** (typically 1-3 seconds)
    
    ### 📊 Model Details
    - **Architecture:** CycleGAN with 6 residual blocks
    - **Training duration:** 50 epochs
    - **Dataset:** Sketchy Database
    - **Input/Output size:** 128x128 pixels
    - **Framework:** PyTorch
    
    ### 💡 Tips for Best Results
    1. Draw clear, well-defined sketches
    2. Use black lines on white background
    3. Avoid too many small details
    4. Keep the sketch centered
    5. For best results, draw faces/objects with clear outlines
    
    ### 🔧 Technical Requirements
    - Modern web browser with JavaScript enabled
    - Stable internet connection for model download (first time only)
    - ~50MB RAM for the model
    
    ### 📧 Need Help?
    If you encounter any issues, please ensure:
    - Your sketch is clear and well-defined
    - You have a stable internet connection
    - Your browser is up to date
    """)

# Sidebar
with st.sidebar:
    st.markdown("## 🎮 Quick Guide")
    st.markdown("""
    ### How to use:
    1. **Choose a method:**
       - ✏️ Draw on canvas
       - 📤 Upload an image
    
    2. **For drawing:**
       - Adjust brush size
       - Pick a color (black works best)
       - Draw your sketch
    
    3. **Click Generate**
    
    4. **Download** your photo!
    
    ### ⚙️ Model Specs
    - **Model:** CycleGAN
    - **Training:** 50 epochs  
    - **Dataset:** Sketchy Database  
    - **Image size:** 128×128
    """)
    
    st.divider()
    
    # Add a fun fact
    st.markdown("### 🎨 Fun Fact")
    st.info("CycleGAN can learn image translation without paired examples! This means it learned to convert sketches to photos just by looking at collections of both types of images.")
    
    # Add a feedback section
    st.divider()
    st.markdown("### 💬 Feedback")
    feedback = st.text_area("Share your thoughts:", placeholder="Loved the app? Have suggestions?")
    if feedback:
        st.success("Thanks for your feedback! 🙏")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Made with ❤️ using Streamlit & CycleGAN | 🚀 Transform your imagination into reality!</p>
    </div>
    """,
    unsafe_allow_html=True
)
