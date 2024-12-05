import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch

def load_model():
    """
    Load YOLOv8 model from best.pt in current directory
    """
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"Fout bij laden van model: {e}")
        return None

def detect_weapons(model, uploaded_file):
    """
    Detect weapons in uploaded image
    """
    # Read image
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Run inference
    results = model(img)
    
    # Process results
    result_img = results[0].plot()
    
    return img, result_img, results[0]

def main():
    st.title("Wapen Detectie met YOLOv8")
    
    # Load model at start
    model = load_model()
    
    if model is None:
        st.error("Kon model niet laden. Controleer of best.pt aanwezig is.")
        return
    
    # Image uploader
    uploaded_image = st.file_uploader(
        "Upload een afbeelding voor wapen detectie", 
        type=['png', 'jpg', 'jpeg']
    )
    
    # Detection logic
    if uploaded_image:
        # Perform detection
        original_img, result_img, detection_results = detect_weapons(model, uploaded_image)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Originele Afbeelding")
            st.image(original_img, channels="BGR")
        
        with col2:
            st.subheader("Detectie Resultaat")
            st.image(result_img, channels="BGR")
        
        # Show detection details
        st.subheader("Detectie Details")
        
        # Display detected objects
        if len(detection_results.boxes) > 0:
            for box in detection_results.boxes:
                # Extract class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]
                
                st.write(f"Gedetecteerd: {label} (Betrouwbaarheid: {conf:.2%})")
        else:
            st.success("Geen wapens gedetecteerd in de afbeelding.")

if __name__ == "__main__":
    # Configureer pagina
    st.set_page_config(
        page_title="Wapen Detectie", 
        page_icon=":gun:", 
        layout="wide"
    )
    
    # Run de app
    main()