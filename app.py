import os
import gdown
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

class CustomRandomRotation(tf.keras.layers.Layer):
    def __init__(self, factor, value_range=None, **kwargs):
        super(CustomRandomRotation, self).__init__(**kwargs)
        self.factor = factor
    def get_config(self):
        config = super(CustomRandomRotation, self).get_config()
        config.update({
            'factor': self.factor,
        })
        return config
    
    def call(self, inputs, training=None):
        return inputs

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the image for model prediction."""
    img = ImageOps.fit(image, target_size, Image.LANCZOS)
    img_array = np.asarray(img)
    
    if len(img_array.shape) > 2 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@st.cache_resource
def load_model():
    model_path = 'best_model.keras'
    
    try:
        if not os.path.exists(model_path):
            with st.spinner('Downloading model from Google Drive... Please wait ⏳'):
                url = 'https://drive.google.com/uc?id=1uMQq0ACeRemwZrPumXcqfaDMSpPjg7eE'
                gdown.download(url, output=model_path, quiet=False)
                st.success("Model downloaded successfully!")
        
        custom_objects = {
            'RandomRotation': CustomRandomRotation,
        }
        
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def predict_tumor(image, model):
    """Make tumor prediction on the image using the loaded model."""
    try:
        processed_img = preprocess_image(image)
        
        predictions = model.predict(processed_img)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class]) * 100
        
        return {
            'class_index': predicted_class,
            'class_name': CLASS_LABELS[predicted_class],
            'confidence': confidence,
            'raw_predictions': predictions[0].tolist()
        }
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

st.set_page_config(
    page_title="Brain Tumor Detector 🧠",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Sidebar content
with st.sidebar:
    st.title("🧠 Brain Tumor Detection")
    st.markdown("""
    ## About
    This tool uses a ResNet50V2-based deep learning model to analyze MRI images
    and detect brain tumors.
    
    ### ⚠️ Medical Disclaimer
    This tool is for educational purposes only and should not replace professional medical advice.
    """)
    
    st.info("Upload an MRI scan to begin.")

# Main app content
st.title("Brain Tumor Detection Tool 🧠")
st.write("Upload an MRI image, and our AI model will predict the presence and type of a brain tumor.")

# File uploader
img_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

# Load model (with spinner to show it's working)
with st.spinner('Loading model...'):
    model = load_model()

if model is None:
    st.error("Failed to load model. Please check the error details above.")
    st.info("Troubleshooting tip: Try restarting the app or check your internet connection.")
    st.stop()

if img_file is None:
    st.warning("Please upload an MRI image to continue.")
    st.subheader("Sample MRI Images")
    st.write("If you don't have an MRI image, you can search for sample brain MRI images online.")
else:
    try:
        image = Image.open(img_file)
        st.image(image, caption="Uploaded MRI Image", use_container_width=True)
        
        with st.spinner('Analyzing image...'):
            result = predict_tumor(image, model)
        
        if result:
            col1, col2 = st.columns(2)
            
            with col1:
                if result['class_name'] == 'notumor':
                    st.success(f"Result: No tumor detected ✅")
                    st.balloons() 
                else:
                    st.error(f"Result: {result['class_name'].capitalize()} tumor detected 🚨")
                
                st.metric("Confidence", f"{result['confidence']:.1f}%")
            
            with col2:
                st.subheader("Analysis Details")
                for label, prob in zip(CLASS_LABELS, result['raw_predictions']):
                    st.progress(float(prob), text=f"{label.capitalize()}: {prob*100:.1f}%")
            
            if result['class_name'] != 'notumor':
                st.warning("⚠️ IMPORTANT: This is an AI-assisted analysis and should not replace professional medical diagnosis. Please consult with a qualified healthcare provider for proper diagnosis and treatment.")
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Try uploading a different image or check if the file is corrupted.")
        
st.sidebar.markdown("---")
st.sidebar.caption("Brain Tumor Detection v1.0")
st.sidebar.caption(f"TensorFlow version: {tf.__version__}")