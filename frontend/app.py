"""
PKD Classification Frontend - Streamlit Application
=====================================================

Modern web interface for kidney CT image classification.
Features image upload, real-time predictions, and result visualization.
"""

import io
import json
import requests
from pathlib import Path
from typing import Optional

import streamlit as st
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="PKD Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS Styling
# ============================================================================

st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1e88e5;
        --secondary-color: #43a047;
        --danger-color: #e53935;
        --warning-color: #fb8c00;
        --background-dark: #0e1117;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Result card styling */
    .result-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    .result-normal {
        border-color: #43a047;
    }
    
    .result-cyst {
        border-color: #fb8c00;
    }
    
    .result-tumor {
        border-color: #e53935;
    }
    
    .result-stone {
        border-color: #7b1fa2;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 0.75rem;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e88e5;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #9e9e9e;
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed #1e88e5;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        background: rgba(30, 136, 229, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        background: rgba(30, 136, 229, 0.1);
        border-color: #42a5f5;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #9e9e9e;
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Configuration
# ============================================================================

API_URL = "http://localhost:8000"

CLASS_COLORS = {
    'Normal': '#43a047',
    'Cyst': '#fb8c00',
    'Tumor': '#e53935',
    'Stone': '#7b1fa2'
}

CLASS_DESCRIPTIONS = {
    'Normal': 'The kidney appears healthy with no visible abnormalities.',
    'Cyst': 'A fluid-filled sac (cyst) detected in the kidney tissue.',
}

CLASS_ICONS = {
    'Normal': '✅',
    'Cyst': '💧',
}

# ============================================================================
# Helper Functions
# ============================================================================

def check_api_health() -> bool:
    """Check if the API is available"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_image(image_bytes: bytes) -> Optional[dict]:
    """Send image to API for prediction"""
    try:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def create_confidence_chart(probabilities: dict) -> go.Figure:
    """Create a horizontal bar chart for confidence scores"""
    classes = list(probabilities.keys())
    values = list(probabilities.values())
    colors = [CLASS_COLORS.get(c, '#1e88e5') for c in classes]
    
    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=classes,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            ),
            text=[f'{v*100:.1f}%' for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=dict(text='Prediction Confidence', font=dict(size=16)),
        xaxis=dict(
            title='Confidence',
            range=[0, 1],
            tickformat='.0%',
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(title='', gridcolor='rgba(255,255,255,0.1)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=300,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig

def create_pie_chart(probabilities: dict) -> go.Figure:
    """Create a pie chart for class distribution"""
    classes = list(probabilities.keys())
    values = list(probabilities.values())
    colors = [CLASS_COLORS.get(c, '#1e88e5') for c in classes]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=classes,
            values=values,
            marker=dict(colors=colors),
            hole=0.4,
            textinfo='label+percent',
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=dict(text='Class Distribution', font=dict(size=16)),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=300,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/kidney.png", width=80)
    st.title("PKD Detection")
    
    st.markdown("---")
    
    # API Status
    st.subheader("🔌 API Status")
    api_status = check_api_health()
    if api_status:
        st.success("✅ Connected")
    else:
        st.error("❌ Disconnected")
        st.info("Start the backend server:\n```bash\ncd backend\nuvicorn main:app --reload\n```")
    
    st.markdown("---")
    
    # Class Information
    st.subheader("📋 Classification Classes")
    for class_name, color in CLASS_COLORS.items():
        icon = CLASS_ICONS[class_name]
        st.markdown(f"{icon} **{class_name}**")
    
    st.markdown("---")
    
    # Model Information
    st.subheader("🤖 Model Info")
    st.markdown("""
    - **Architecture:** EfficientNet-B0
    - **Input Size:** 224×224
    - **Classes:** 4
    - **Training:** 5-Fold CV
    """)
    
    st.markdown("---")
    
    # Credits
    st.subheader("👥 Team")
    st.markdown("""
    - Aniket Waghela
    - Bhavya Jappi
    - Kerul Kidecha
    
    **Guide:** Prof. Kanchan Dabre
    """)

# ============================================================================
# Main Content
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 PKD Detection System</h1>
    <p>AI-Powered Kidney CT Scan Classification</p>
</div>
""", unsafe_allow_html=True)

# Upload Section
st.subheader("📤 Upload CT Scan Image")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a kidney CT scan image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear CT scan image of the kidney"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Reset file position
        uploaded_file.seek(0)
        image_bytes = uploaded_file.read()

with col2:
    if uploaded_file is not None:
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            if not api_status:
                st.error("API is not available. Please start the backend server.")
            else:
                with st.spinner("Analyzing image..."):
                    result = predict_image(image_bytes)
                
                if result:
                    # Store result in session state
                    st.session_state['last_result'] = result
                    st.session_state['has_result'] = True
                    st.rerun()
    else:
        st.info("👆 Please upload a CT scan image to begin analysis")
        
        # Sample images section
        st.markdown("### 📸 Sample Images")
        st.markdown("""
        For best results, upload:
        - Clear CT scan images
        - Kidney-focused views
        - JPG/PNG format
        """)

# Results Section
if st.session_state.get('has_result', False):
    result = st.session_state['last_result']
    
    st.markdown("---")
    st.subheader("📊 Analysis Results")
    
    predicted_class = result['class_name']
    confidence = result['confidence']
    probabilities = result['all_probabilities']
    
    # Result card
    result_class = predicted_class.lower()
    icon = CLASS_ICONS[predicted_class]
    description = CLASS_DESCRIPTIONS[predicted_class]
    color = CLASS_COLORS[predicted_class]
    
    st.markdown(f"""
    <div class="result-card result-{result_class}">
        <h2 style="color: {color}; margin-bottom: 0.5rem;">{icon} {predicted_class}</h2>
        <p style="color: #e0e0e0; margin-bottom: 1rem;">{description}</p>
        <div class="metric-card" style="display: inline-block; padding: 0.5rem 1rem;">
            <span class="metric-value" style="color: {color};">{confidence*100:.1f}%</span>
            <span class="metric-label"> Confidence</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        bar_chart = create_confidence_chart(probabilities)
        st.plotly_chart(bar_chart, use_container_width=True)
    
    with chart_col2:
        pie_chart = create_pie_chart(probabilities)
        st.plotly_chart(pie_chart, use_container_width=True)
    
    # Detailed probabilities
    with st.expander("📈 Detailed Probabilities"):
        for class_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(prob)
            with col2:
                st.write(f"**{class_name}:** {prob*100:.2f}%")
    
    # Disclaimer
    st.warning("""
    ⚠️ **Medical Disclaimer:** This AI-powered analysis is intended for educational and 
    research purposes only. It should not be used as a substitute for professional 
    medical diagnosis. Please consult a qualified healthcare provider for medical advice.
    """)
    
    # Clear results button
    if st.button("🔄 Analyze Another Image"):
        st.session_state['has_result'] = False
        st.session_state['last_result'] = None
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>PKD Detection System | Department of Computer Science and Engineering (Data Science)</p>
    <p>Powered by EfficientNet-B0 | Built with Streamlit & FastAPI</p>
</div>
""", unsafe_allow_html=True)
