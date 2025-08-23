import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tempfile
import os
from PIL import Image
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="BloodCell AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff4b2b;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(45deg, #ff4b2b, #ff6b3d);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-box {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and class labels
@st.cache_resource
def load_blood_cell_model():
    try:
        # Try to load the model from the blood_cell_model.h5 folder
        model_path = "blood_cell_model.h5/blood_cell_model.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
            return model
        else:
            st.error("Model file not found. Please ensure blood_cell_model.h5 exists in the project directory.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Class labels
labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

# Load model
model = load_blood_cell_model()

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 BloodCell AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Blood Cell Classification using Transfer Learning</p>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    This application uses deep learning and transfer learning techniques to accurately classify human blood cells into different categories:
    - **EOSINOPHIL** - Allergic reactions and parasitic infections
    - **LYMPHOCYTE** - Viral infections and immune responses
    - **MONOCYTE** - Chronic infections and inflammation
    - **NEUTROPHIL** - Bacterial infections and acute inflammation
    """)
    
    st.markdown("---")
    
    # File upload section
    st.header("📸 Upload Blood Cell Image")
    st.markdown("Upload a blood cell image to classify it into one of the four types.")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        st.subheader("Uploaded Image")
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption="Uploaded Blood Cell Image", use_column_width=True)
        
        # Prediction button
        if st.button("🔬 Analyze Blood Cell", type="primary"):
            if model is not None:
                with st.spinner("Analyzing blood cell..."):
                    try:
                        # Preprocess image
                        img = image_display.resize((224, 224))
                        img_array = np.array(img)
                        img_array = img_array / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        # Make prediction
                        prediction = model.predict(img_array)
                        class_index = np.argmax(prediction)
                        class_name = labels[class_index]
                        confidence = float(prediction[0][class_index])
                        
                        # Display results
                        st.success("Analysis Complete!")
                        
                        # Prediction result
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>Predicted Cell Type: {class_name}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence score
                        st.markdown(f"""
                        <div class="confidence-box">
                            <h3>Confidence: {confidence:.2%}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Cell type description
                        cell_descriptions = {
                            'EOSINOPHIL': 'Eosinophils are white blood cells that help fight parasitic infections and are involved in allergic reactions.',
                            'LYMPHOCYTE': 'Lymphocytes are white blood cells that help fight viral infections and are key components of the immune system.',
                            'MONOCYTE': 'Monocytes are white blood cells that help fight chronic infections and are involved in inflammatory responses.',
                            'NEUTROPHIL': 'Neutrophils are the most common white blood cells and are the first responders to bacterial infections.'
                        }
                        
                        st.info(f"**About {class_name}s:** {cell_descriptions[class_name]}")
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
            else:
                st.error("Model not loaded. Please check the model file.")
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About BloodCell AI")
        st.markdown("""
        **BloodCell AI** is an advanced machine learning application that uses transfer learning to classify blood cells.
        
        **How it works:**
        1. Upload a blood cell image
        2. Our AI model analyzes the image
        3. Get instant classification results
        4. View confidence scores and descriptions
        
        **Use cases:**
        - Medical research
        - Laboratory automation
        - Educational purposes
        - Clinical decision support
        
        **Note:** This tool is for educational and research purposes. Always consult healthcare professionals for medical decisions.
        """)
        
        st.markdown("---")
        st.markdown("**🔬 Built with:**")
        st.markdown("- TensorFlow/Keras")
        st.markdown("- Transfer Learning")
        st.markdown("- Streamlit")
        st.markdown("- Python")

if __name__ == "__main__":
    main()
