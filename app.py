import streamlit as st
import os
import sys
import numpy as np
from PIL import Image
import pickle

# Set page config
st.set_page_config(page_title="Age Predictor", layout="wide")

# Import ML packages with error handling
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    import xgboost as xgb
except ImportError as e:
    st.error(f"Error importing ML packages: {str(e)}")
    st.error("Please check if all required packages are installed:")
    st.code("pip install -r requirements.txt")
    st.stop()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.info(f"Using device: {device}")

# Show environment information
st.write("Python version:", sys.version)
st.write("Python path:", sys.path)
st.write("Current working directory:", os.getcwd())
st.write("Available files:", os.listdir())

# Load models and preprocessing objects
@st.cache_resource
def load_models():
    try:
        # Load face model
        st.write("Attempting to load face model...")
        
        model = models.resnext50_32x4d(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        
        model_path = os.path.join(os.getcwd(), 'face_model.pth')
        st.write(f"Looking for face model at: {model_path}")
        if not os.path.exists(model_path):
            st.error(f"Face model not found at {model_path}")
            raise FileNotFoundError(f"Face model not found at {model_path}")
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        st.write("Face model loaded successfully")

        # Load XGBoost models with error checking
        st.write("Loading XGBoost models...")
        bio_model = xgb.XGBRegressor()
        bio_model_path = os.path.join(os.getcwd(), 'bio_model.json')
        if not os.path.exists(bio_model_path):
            st.error(f"Bio model not found at {bio_model_path}")
            raise FileNotFoundError(f"Bio model not found at {bio_model_path}")
        bio_model.load_model(bio_model_path)
        st.write("Bio model loaded successfully")
        
        face_adjuster = xgb.XGBRegressor()
        face_adj_path = os.path.join(os.getcwd(), 'face_adjuster.json')
        if not os.path.exists(face_adj_path):
            st.error(f"Face adjuster model not found at {face_adj_path}")
            raise FileNotFoundError(f"Face adjuster model not found at {face_adj_path}")
        face_adjuster.load_model(face_adj_path)
        st.write("Face adjuster loaded successfully")
        
        stacker = xgb.XGBRegressor()
        stacker_path = os.path.join(os.getcwd(), 'stacker.json')
        if not os.path.exists(stacker_path):
            st.error(f"Stacker model not found at {stacker_path}")
            raise FileNotFoundError(f"Stacker model not found at {stacker_path}")
        stacker.load_model(stacker_path)
        st.write("Stacker model loaded successfully")

        # Load preprocessing objects with error checking
        st.write("Loading preprocessing objects...")
        scaler_path = os.path.join(os.getcwd(), 'scaler.pkl')
        poly_path = os.path.join(os.getcwd(), 'poly.pkl')
        selector_path = os.path.join(os.getcwd(), 'selector.pkl')
        
        if not all(os.path.exists(p) for p in [scaler_path, poly_path, selector_path]):
            missing = [p for p in [scaler_path, poly_path, selector_path] if not os.path.exists(p)]
            st.error(f"Missing preprocessing files: {missing}")
            raise FileNotFoundError(f"Missing preprocessing files: {missing}")

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(poly_path, 'rb') as f:
            poly = pickle.load(f)
        with open(selector_path, 'rb') as f:
            selector = pickle.load(f)
        st.write("All preprocessing objects loaded successfully")

        return model, bio_model, face_adjuster, stacker, scaler, poly, selector, device
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Stack trace:", exc_info=True)
        raise e

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

# Preprocess biomarker features
def preprocess_biomarkers(features, scaler, poly, selector):
    features_poly = poly.transform(features.reshape(1, -1))
    features_scaled = scaler.transform(features_poly)
    features_selected = selector.transform(features_scaled)
    return features_selected

def main():
    st.title("Age Prediction from Face and Biomarkers")
    st.write("Upload a face image and enter biomarker values to predict age")

    try:
        model, bio_model, face_adjuster, stacker, scaler, poly, selector, device = load_models()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Face Image")
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption='Uploaded Image', use_column_width=True)
        
        with col2:
            st.subheader("Biomarkers")
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
            gender = st.selectbox("Gender", ["Male", "Female"])
            ap_hi = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120)
            ap_lo = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=150, value=80)
            
            # New cholesterol and glucose inputs with numerical values
            st.write("Cholesterol Level (mg/dL)")
            cholesterol_type = st.radio("Input type for Cholesterol", ["Categories", "Specific Value"], horizontal=True)
            if cholesterol_type == "Categories":
                cholesterol = st.selectbox("Cholesterol Category", [1, 2, 3], 
                                     format_func=lambda x: {1: "Normal (<200)", 2: "Above Normal (200-239)", 3: "High (≥240)"}[x])
            else:
                cholesterol_value = st.number_input("Cholesterol Value (mg/dL)", min_value=100, max_value=500, value=200)
                # Convert specific value to category
                if cholesterol_value < 200:
                    cholesterol = 1
                elif cholesterol_value < 240:
                    cholesterol = 2
                else:
                    cholesterol = 3
            
            st.write("Glucose Level (mg/dL)")
            glucose_type = st.radio("Input type for Glucose", ["Categories", "Specific Value"], horizontal=True)
            if glucose_type == "Categories":
                glucose = st.selectbox("Glucose Category", [1, 2, 3],
                                 format_func=lambda x: {1: "Normal (<100)", 2: "Above Normal (100-125)", 3: "High (≥126)"}[x])
            else:
                glucose_value = st.number_input("Glucose Value (mg/dL)", min_value=50, max_value=300, value=100)
                # Convert specific value to category
                if glucose_value < 100:
                    glucose = 1
                elif glucose_value < 126:
                    glucose = 2
                else:
                    glucose = 3

        if st.button("Predict Age") and uploaded_file is not None:
            with st.spinner("Processing..."):
                # Process face image
                image_tensor = preprocess_image(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    face_pred = model(image_tensor).cpu().numpy().flatten()[0]

                # Process biomarkers
                gender_val = 0 if gender == "Male" else 1
                biomarker_features = np.array([height, weight, gender_val, ap_hi, ap_lo, cholesterol, glucose])
                
                # Calculate derived features
                bmi = weight / ((height / 100) ** 2)
                bp_diff = ap_hi - ap_lo
                bp_mean = (ap_hi + ap_lo) / 2
                cholesterol_gluc_interaction = cholesterol * glucose
                bmi_bp_interaction = bmi * bp_mean
                
                features = np.array([
                    height, weight, gender_val, ap_hi, ap_lo, cholesterol, glucose,
                    bmi, bp_diff, bp_mean, cholesterol_gluc_interaction, bmi_bp_interaction
                ])
                
                bio_features = preprocess_biomarkers(features, scaler, poly, selector)
                bio_pred = bio_model.predict(bio_features)[0]

                # Adjust face prediction
                stack_X_face = np.column_stack((face_pred.reshape(1, -1), bio_features))
                face_pred_adj = face_adjuster.predict(stack_X_face)[0]

                # Final stacking prediction
                stack_X_hybrid = np.column_stack((face_pred_adj.reshape(1, -1), bio_pred.reshape(1, -1)))
                final_pred = stacker.predict(stack_X_hybrid)[0]

                # Display results
                st.success("Prediction Complete!")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Face-based Age", f"{face_pred_adj:.1f} years")
                with col2:
                    st.metric("Biomarker-based Age", f"{bio_pred:.1f} years")
                with col3:
                    st.metric("Combined Prediction", f"{final_pred:.1f} years")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure all model files are in the correct location and try again.")

if __name__ == "__main__":
    main()
