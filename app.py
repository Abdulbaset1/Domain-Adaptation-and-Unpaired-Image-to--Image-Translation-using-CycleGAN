# app.py
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import requests
import os
import tempfile
from io import BytesIO
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="CycleGAN Image Translation",
    page_icon="🎨",
    layout="wide"
)

# Model URL from GitHub release
MODEL_URL = "https://github.com/Abdulbaset1/Domain-Adaptation-and-Unpaired-Image-to--Image-Translation-using-CycleGAN/releases/download/v1/cyclegan_model.pth"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture (same as training)
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        layers = [
            nn.Conv2d(3, 64, 7, 1, 3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Downsample
        in_c = 64
        for _ in range(2):
            layers += [
                nn.Conv2d(in_c, in_c*2, 3, 2, 1),
                nn.InstanceNorm2d(in_c*2),
                nn.ReLU(inplace=True)
            ]
            in_c *= 2
        
        # 6 ResBlocks
        for _ in range(6):
            layers.append(ResBlock(in_c))
        
        # Upsample
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(in_c, in_c//2, 3, 2, 1, output_padding=1),
                nn.InstanceNorm2d(in_c//2),
                nn.ReLU(inplace=True)
            ]
            in_c //= 2
        
        layers += [
            nn.Conv2d(in_c, 3, 7, 1, 3),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model():
    """Load the CycleGAN model from GitHub release"""
    try:
        # Create a temporary file to store the model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            st.info("📥 Downloading model from GitHub release...")
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            
            # Download with progress bar
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = st.progress(0)
            downloaded = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = downloaded / total_size
                    progress_bar.progress(min(progress, 1.0))
            
            progress_bar.empty()
            tmp_file_path = tmp_file.name
        
        # Load the model
        st.info("🔧 Loading model architecture...")
        model = Generator().to(DEVICE)
        
        st.info("📦 Loading model weights...")
        state_dict = torch.load(tmp_file_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        st.success("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def transform_image(image, target_size=256):
    """Transform input image for the model"""
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor.to(DEVICE)

def denormalize(tensor):
    """Denormalize tensor to displayable image"""
    return tensor * 0.5 + 0.5

def translate_image(model, image_tensor):
    """Translate image using the model"""
    with torch.no_grad():
        output = model(image_tensor)
    return denormalize(output)

def tensor_to_pil(tensor):
    """Convert tensor to PIL image"""
    tensor = tensor.squeeze(0).cpu()
    tensor = tensor.clamp(0, 1)
    img = transforms.ToPILImage()(tensor)
    return img

# Main UI
st.title("🎨 CycleGAN: Sketch to Photo Translation")
st.markdown("""
This application uses a CycleGAN model to translate between **sketches and photos**.
Upload an image and choose the translation direction!
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    translation_direction = st.radio(
        "Select translation direction:",
        ["Sketch → Photo", "Photo → Sketch"],
        help="Choose which transformation you want to apply"
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    - **Model**: CycleGAN trained on sketches and photos
    - **Architecture**: Generator with ResNet blocks
    - **Input Size**: 256x256 pixels
    """)
    
    st.markdown("---")
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. Upload an image (JPG, PNG)
    2. Select translation direction
    3. Wait for processing
    4. Download the result
    """)

# Main content area
col1, col2 = st.columns(2)

# Load model
model = load_model()

if model is None:
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
    help="Upload an image to translate"
)

if uploaded_file is not None:
    # Load and display original image
    original_image = Image.open(uploaded_file)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(original_image, use_container_width=True)
        
        # Display image info
        st.caption(f"Size: {original_image.size[0]} x {original_image.size[1]} pixels")
        st.caption(f"Mode: {original_image.mode}")
    
    # Transform and translate
    with st.spinner("🔄 Processing image... This may take a few seconds."):
        try:
            # Transform image
            input_tensor = transform_image(original_image)
            
            # Translate
            output_tensor = translate_image(model, input_tensor)
            
            # Convert to PIL
            output_image = tensor_to_pil(output_tensor)
            
            # Display output
            with col2:
                if translation_direction == "Sketch → Photo":
                    st.subheader("🎨 Generated Photo")
                else:
                    st.subheader("✏️ Generated Sketch")
                st.image(output_image, use_container_width=True)
                
                # Download button
                buf = BytesIO()
                output_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Download Result",
                    data=byte_im,
                    file_name=f"translated_{uploaded_file.name.split('.')[0]}.png",
                    mime="image/png"
                )
                
        except Exception as e:
            st.error(f"Error during translation: {str(e)}")
else:
    # Show example when no image is uploaded
    st.info("👈 Please upload an image to get started!")
    
    # Display example images
    st.markdown("### 📸 Example Usage")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Input Sketch** →")
        st.markdown("*(Upload a sketch to convert to photo)*")
    
    with col2:
        st.markdown("**Input Photo** →")
        st.markdown("*(Upload a photo to convert to sketch)*")
    
    with col3:
        st.markdown("**⚡ Features**")
        st.markdown("""
        - Real-time translation
        - High-quality results
        - Maintains structural consistency
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using CycleGAN | Model trained on sketches and CIFAR-10 photos</p>
    </div>
    """,
    unsafe_allow_html=True
)
