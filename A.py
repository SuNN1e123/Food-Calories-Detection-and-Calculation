import streamlit as st
import numpy as np
from PIL import Image
import requests
import time
import io
from datetime import datetime

# Check for required packages and provide installation instructions if missing
try:
    import cv2
except ImportError:
    st.error("""
    **OpenCV (cv2) is not installed.**  
    Please install it by running:  
    `pip install opencv-python-headless`
    """)
    st.stop()

try:
    import tensorflow as tf
except ImportError:
    st.error("""
    **TensorFlow is not installed.**  
    Please install it by running:  
    `pip install tensorflow`
    """)
    st.stop()

# Mock food recognition model (simplified version)
class FoodDetector:
    def __init__(self):
        self.labels = [
            "Apple", "Banana", "Burger", "Chocolate", 
            "Chocolate Donut", "French Fries", "Fruit Oatmeal",
            "Pear", "Potato Chips", "Rice"
        ]
        self.food_db = {
            "Apple": {"calories": 52, "healthy": True},
            "Banana": {"calories": 89, "healthy": True},
            "Burger": {"calories": 313, "healthy": False},
            "Chocolate": {"calories": 535, "healthy": False},
            "Chocolate Donut": {"calories": 452, "healthy": False},
            "French Fries": {"calories": 312, "healthy": False},
            "Fruit Oatmeal": {"calories": 68, "healthy": True},
            "Pear": {"calories": 57, "healthy": True},
            "Potato Chips": {"calories": 536, "healthy": False},
            "Rice": {"calories": 130, "healthy": True}
        }
    
    def detect_food(self, image):
        """Mock detection function - replace with your actual model"""
        # In a real app, you would use your TensorFlow model here
        # For demo purposes, we'll return a random food item
        import random
        food = random.choice(self.labels)
        return food, random.uniform(0.7, 0.99)  # Random confidence

# Initialize the detector
detector = FoodDetector()

# Streamlit app layout
st.set_page_config(page_title="Food Calorie Calculator", layout="wide")

# Title and description
st.title("🍏 Food Calorie Calculator")
st.markdown("""
Capture an image of your food to detect what it is and calculate its nutritional information.
""")

# Create two main columns
col1, col2 = st.columns([1, 1], gap="large")

# Column 1: Image Capture and Detection
with col1:
    st.header("Food Detection")
    
    # Image upload option (simplified)
    uploaded_file = st.file_uploader("Upload a food image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Food Image", use_column_width=True)
        
        if st.button("Detect Food"):
            with st.spinner("Detecting food..."):
                # Convert PIL Image to OpenCV format
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Detect food (using mock detector)
                detected_food, confidence = detector.detect_food(opencv_image)
                
                if detected_food:
                    st.success(f"Detected: {detected_food} (Confidence: {confidence:.1%})")
                    st.session_state.detected_food = detected_food
                    st.session_state.food_image = image
                else:
                    st.warning("No food item detected")

# Column 2: Portion Input and Results
with col2:
    if 'detected_food' in st.session_state:
        st.header("Nutritional Information")
        
        # Display detected food
        st.subheader("Detected Food")
        col_img, col_info = st.columns([1, 2])
        with col_img:
            st.image(st.session_state.food_image, width=150)
        with col_info:
            st.markdown(f"**{st.session_state.detected_food}**")
            food_data = detector.food_db.get(st.session_state.detected_food)
            if food_data['healthy']:
                st.success("✅ Healthy food")
            else:
                st.warning("⚠️ Not recommended for frequent consumption")
        
        # Portion input
        st.subheader("Portion Details")
        portion_option = st.selectbox(
            "Select portion size:",
            ["Custom", "Small (50g)", "Medium (100g)", "Large (150g)"],
            index=1
        )
        
        if portion_option == "Custom":
            weight = st.number_input("Enter weight (grams):", min_value=1, value=100)
        elif portion_option == "Small (50g)":
            weight = 50
        elif portion_option == "Medium (100g)":
            weight = 100
        else:  # Large
            weight = 150
        
        # Calculate calories
        if st.button("Calculate Nutrition"):
            food_data = detector.food_db.get(st.session_state.detected_food)
            if food_data:
                calories = (food_data['calories'] * weight) / 100
                
                # Display results
                st.subheader("Nutritional Information")
                
                cols = st.columns(2)
                cols[0].metric("Food", st.session_state.detected_food)
                cols[1].metric("Weight", f"{weight}g")
                
                cols = st.columns(2)
                cols[0].metric("Calories", f"{calories:.1f} kcal")
                cols[1].metric("Calories per 100g", f"{food_data['calories']} kcal")
            else:
                st.error("Nutritional data not available for this food item")
    else:
        st.header("Nutritional Information")
        st.info("Please detect a food item first to see nutritional information")

# Footer
st.markdown("---")
st.markdown("""
**How to use:**
1. Upload an image of your food
2. The system will detect what food it is
3. Select the portion size
4. View the calculated nutritional information
""")
