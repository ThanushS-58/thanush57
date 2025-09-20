#!/usr/bin/env python3
"""
Pre-trained sklearn model integration for medicinal plant classification
Uses the uploaded Best_Model.pkl, pca.pkl, and scaler.pkl files
"""

import pickle
import numpy as np
import os
import sys
from PIL import Image
import cv2
import json

class SklearnPlantClassifier:
    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        self.model = None
        self.pca = None
        self.scaler = None
        self.class_names = [
            'Aloevera', 'Amla', 'Amruta_Balli', 'Arali', 'Ashoka', 'Ashwagandha',
            'Avacado', 'Banana', 'Basale', 'Betel', 'Betel_Nut', 'Brahmi',
            'Castor', 'Catharanthus', 'Chakte', 'Chilly', 'Citron_lime',
            'Coffee', 'Commonrue', 'Coriander', 'Curry', 'Doddpathre',
            'Drumstick', 'Ekka', 'Eucalyptus', 'Ganike', 'Ganuga', 'Gasagase',
            'Ginger', 'Globe_Amarnath', 'Guava', 'Henna', 'Hibiscus', 'Honge',
            'Insulin', 'Jackfruit', 'Jasmine', 'Kambajala', 'Kasambruga',
            'Kohlrabi', 'Lantana', 'Lemon', 'Lemongrass', 'Mango', 'Marigold',
            'Mint', 'Neem', 'Nelavembu', 'Nerale', 'Nooni', 'Onion', 'Padri',
            'Palak(Spinach)', 'Papaya', 'Parijatha', 'Pea', 'Pepper',
            'Pomegranate', 'Pumpkin', 'Raddish', 'Rose', 'Rosemary', 'Sapota',
            'Seethaashoka', 'Seethapala', 'Spinach1', 'Tamarind', 'Taro',
            'Tecoma', 'Thumbe', 'Tomato', 'Tulsi', 'Turmeric', 'Wood_sorel'
        ]
        self.load_models()
    
    def load_models(self):
        """Load the pre-trained models"""
        try:
            # Load main classifier
            model_path = os.path.join(self.models_dir, 'Best_Model.pkl')
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load PCA
            pca_path = os.path.join(self.models_dir, 'pca.pkl')
            with open(pca_path, 'rb') as f:
                self.pca = pickle.load(f)
            
            # Load scaler
            scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            print("✅ Pre-trained sklearn models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.model = None
    
    def extract_features(self, image_path):
        """Extract features from image using OpenCV"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image")
            
            # Resize to standard size
            img = cv2.resize(img, (128, 128))
            
            # Convert to different color spaces and extract features
            features = []
            
            # RGB features
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            features.extend(rgb_img.flatten())
            
            # HSV features
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            features.extend(hsv_img.mean(axis=(0,1)))
            features.extend(hsv_img.std(axis=(0,1)))
            
            # LAB features
            lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            features.extend(lab_img.mean(axis=(0,1)))
            features.extend(lab_img.std(axis=(0,1)))
            
            # Texture features using Sobel
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            features.extend([sobelx.mean(), sobelx.std(), sobely.mean(), sobely.std()])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"❌ Feature extraction error: {e}")
            return None
    
    def classify_image(self, image_path):
        """Classify plant image using pre-trained model"""
        if self.model is None:
            return None
        
        try:
            # Extract features
            features = self.extract_features(image_path)
            if features is None:
                return None
            
            # Normalize features
            features_scaled = self.scaler.transform(features)
            
            # Apply PCA
            features_pca = self.pca.transform(features_scaled)
            
            # Get prediction and probabilities
            prediction = self.model.predict(features_pca)[0]
            probabilities = self.model.predict_proba(features_pca)[0]
            
            # Get top 3 predictions
            top_indices = np.argsort(probabilities)[-3:][::-1]
            
            result = {
                'predicted_class': self.class_names[prediction],
                'confidence': float(probabilities[prediction] * 100),
                'top_predictions': [
                    {
                        'plant': self.class_names[idx],
                        'confidence': float(probabilities[idx] * 100)
                    }
                    for idx in top_indices
                ]
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Classification error: {e}")
            return None

def main():
    """Command line interface for testing"""
    if len(sys.argv) != 2:
        print("Usage: python sklearn-plant-classifier.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Initialize classifier
    classifier = SklearnPlantClassifier()
    
    # Classify image
    result = classifier.classify_image(image_path)
    
    if result:
        print(json.dumps(result, indent=2))
    else:
        print("❌ Classification failed")

if __name__ == "__main__":
    main()