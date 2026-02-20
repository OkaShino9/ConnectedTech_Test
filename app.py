import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from PIL import Image

def main():
    st.set_page_config(page_title="Bottle Detection", page_icon="💧", layout="wide")
    st.title("Blue Water Bottle Detection & Counting")
    st.markdown("### AI Engineer Selection Test")
    st.write("This application detects and counts blue water bottles inside a plastic bag using a trained YOLOv8 model.")

    # Load model
    model_path = "best11s.pt"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at `{model_path}`. Please ensure the model is in the correct directory.")
        return

    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Directory where images are expected
    st.sidebar.title("Simulation Options")
    st.sidebar.info("Select one of the test images to verify the model's performance.")
    
    # Use images present in the root
    image_files = ["Data_Bottles.png", "Example_Test_Bottles.png"]
    
    selected_image = st.sidebar.selectbox("Select Image for Detection", image_files)
    
    if selected_image:
        img_path = selected_image
        if os.path.exists(img_path):
            image = Image.open(img_path)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
                
            with col2:
                st.subheader("Action")
                if st.button("Detect and Count Bottles", type="primary"):
                    with st.spinner('Detecting objects...'):
                        # Preprocess image
                        cv_image = cv2.imread(img_path)
                        # Switch to Colored Denoising for better local detection accuracy
                        denoised_image = cv2.fastNlMeansDenoisingColored(cv_image)
                        auto_contrast = cv2.normalize(denoised_image, None, 0, 175, cv2.NORM_MINMAX)
                        smoothed_image = cv2.GaussianBlur(auto_contrast, (5, 5), 0)

                        # Run YOLO prediction (tuned for Streamlit/Local Hardware execution)
                        results = model.predict(smoothed_image, imgsz=1280, conf=0.1, iou=0.1, save=False)
                        
                        # Plot the results on the image
                        result_img_array = results[0].plot()
                        # Convert BGR (cv2/ultralytics) to RGB for Streamlit/PIL
                        result_image = Image.fromarray(result_img_array[..., ::-1])
                        
                        # Count total bounding boxes detected
                        count = len(results[0].boxes)
                        
                        st.subheader("Detection Result")
                        st.image(result_image, caption=f"Detected Bottles: {count}", use_container_width=True)
                        if count > 0:
                            st.success(f"Successfully detected {count} blue water bottles!")
                        else:
                            st.warning("No bottles detected.")
        else:
            st.error(f"Image `{selected_image}` not found in the current directory.")

if __name__ == "__main__":
    main()
