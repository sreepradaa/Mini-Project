import streamlit as st
import cv2
import os
from process import process_input
from PIL import Image
import numpy as np
import base64
from io import BytesIO  # Added import for BytesIO

# Streamlit page configuration
st.set_page_config(page_title="Traffic Violation Detection System", layout="wide")

# Create violations folder if it doesn't exist
VIOLATION_FOLDER = 'violations/'
if not os.path.exists(VIOLATION_FOLDER):
    os.makedirs(VIOLATION_FOLDER)

def main():
    st.title("🚦 Traffic Violation Detection System")
    st.markdown("Detect motorcycles without helmets and cars without seatbelts in real-time using YOLOv8 and ANPR.")

    # Sidebar for input selection
    st.sidebar.header("Input Options")
    input_type = st.sidebar.radio("Select Input Type", ["Image", "Video", "Live Webcam"])

    # Main content
    if input_type == "Image":
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Save uploaded file
            file_path = os.path.join("temp_image.jpg")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process image
            result = process_input(file_path, is_video=False)
            
            # Display results
            st.subheader("Processed Image")
            for frame_base64 in result['frames']:
                frame = Image.open(BytesIO(base64.b64decode(frame_base64)))
                # Updated code
                st.image(frame, caption="Processed Frame", use_container_width=True)
            
            # Display violations
            st.subheader("Violations Detected")
            if result['violations']:
                for violation in result['violations']:
                    st.write(f"**Type**: {violation['type']}, **Confidence**: {violation['confidence']:.2f}, **Time**: {violation['timestamp']}")
            else:
                st.write("No violations detected.")

    elif input_type == "Video":
        uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi"])
        if uploaded_file is not None:
            # Save uploaded file
            file_path = os.path.join("temp_video.mp4")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process video
            result = process_input(file_path, is_video=True)
            
            # Display results
            st.subheader("Processed Frames")
            cols = st.columns(3)
            for i, frame_base64 in enumerate(result['frames']):
                frame = Image.open(BytesIO(base64.b64decode(frame_base64)))
                cols[i % 3].image(frame, use_container_width=True)
            
            # Display violations
            st.subheader("Violations Detected")
            if result['violations']:
                for violation in result['violations']:
                    st.write(f"**Type**: {violation['type']}, **Confidence**: {violation['confidence']:.2f}, **Time**: {violation['timestamp']}")
            else:
                st.write("No violations detected.")

    elif input_type == "Live Webcam":
        st.warning("Ensure your webcam is connected.")
        if st.button("Start Live Feed"):
            # Process live feed
            result = process_input(0, is_video=True)
            
            # Display results
            st.subheader("Live Feed Frames")
            cols = st.columns(3)
            for i, frame_base64 in enumerate(result['frames']):
                frame = Image.open(BytesIO(base64.b64decode(frame_base64)))
                cols[i % 3].image(frame, use_container_width=True)
            
            # Display violations
            st.subheader("Violations Detected")
            if result['violations']:
                for violation in result['violations']:
                    st.write(f"**Type**: {violation['type']}, **Confidence**: {violation['confidence']:.2f}, **Time**: {violation['timestamp']}")
            else:
                st.write("No violations detected.")

    # Violations Log
    st.sidebar.header("Violations Log")
    violation_files = [f for f in os.listdir(VIOLATION_FOLDER) if f.endswith('.jpg')]
    if violation_files:
        for file in violation_files:
            st.sidebar.image(os.path.join(VIOLATION_FOLDER, file), caption=file.replace('.jpg', ''), width=200)
    else:
        st.sidebar.write("No violations logged yet.")

if __name__ == "__main__":
    main()