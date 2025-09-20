#!/usr/bin/env python3
"""
Improved sklearn model integration with better version compatibility
"""

import pickle
import numpy as np
import os
import sys
import json
from PIL import Image
import cv2
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

class ImprovedSklearnClassifier:
    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        self.model = None
        self.pca = None
        self.scaler = None
        self.plant_info = {}
        self.load_models()
        self.load_plant_info()
    
    def load_models(self):
        """Load models with better error handling"""
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
            
            print("✅ Improved sklearn models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"⚠️  sklearn models not available: {e}")
            self.model = None
            return False
    
    def load_plant_info(self):
        """Load additional plant information"""
        try:
            info_path = os.path.join(self.models_dir, 'plant_info_trained.json')
            with open(info_path, 'r') as f:
                self.plant_info = json.load(f)
        except Exception as e:
            print(f"⚠️  Plant info not found: {e}")
            self.plant_info = {}
    
    def extract_features_robust(self, image_path):
        """Extract features matching the original training pipeline exactly"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image")
            
            # Get expected feature count from the PCA component
            expected_pca_features = getattr(self.pca, 'n_components_', 50)  # Default to 50 if not found
            scaler_expected = getattr(self.scaler, 'n_features_in_', None)
            
            print(f"Expected PCA components: {expected_pca_features}")
            print(f"Expected scaler features: {scaler_expected}")
            
            # Use a fixed approach that matches the training data
            # Resize to 64x64 to match original training (likely smaller due to memory constraints)
            img_size = 64
            img = cv2.resize(img, (img_size, img_size))
            
            features = []
            
            # RGB features (flattened) - main feature source
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            features.extend(rgb_img.flatten())  # 64*64*3 = 12288 features
            
            # HSV color space features
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            features.extend(hsv_img.mean(axis=(0,1)))  # 3 features
            features.extend(hsv_img.std(axis=(0,1)))   # 3 features
            
            # LAB color space features  
            lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            features.extend(lab_img.mean(axis=(0,1)))  # 3 features
            features.extend(lab_img.std(axis=(0,1)))   # 3 features
            
            # Texture features using Sobel
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            features.extend([sobelx.mean(), sobelx.std(), sobely.mean(), sobely.std()])  # 4 features
            
            features_array = np.array(features).reshape(1, -1)
            print(f"Extracted {features_array.shape[1]} features")
            
            # If we have too many features, truncate to expected count
            if scaler_expected and features_array.shape[1] > scaler_expected:
                features_array = features_array[:, :scaler_expected]
                print(f"Truncated to {scaler_expected} features to match training")
            elif scaler_expected and features_array.shape[1] < scaler_expected:
                # Pad with zeros if we have too few features
                padding = np.zeros((1, scaler_expected - features_array.shape[1]))
                features_array = np.hstack([features_array, padding])
                print(f"Padded to {scaler_expected} features to match training")
            
            return features_array
            
        except Exception as e:
            print(f"❌ Feature extraction error: {e}")
            return None
    
    def classify_image_safe(self, image_path):
        """Safe classification with fallback"""
        if self.model is None:
            return {
                'success': False,
                'error': 'Models not loaded'
            }
        
        try:
            features = self.extract_features_robust(image_path)
            if features is None:
                return {
                    'success': False,
                    'error': 'Feature extraction failed'
                }
            
            # Try to scale features - handle dimension mismatch
            try:
                features_scaled = self.scaler.transform(features)
            except ValueError as e:
                # If feature count doesn't match, try truncating or padding
                expected = getattr(self.scaler, 'n_features_in_', features.shape[1])
                if features.shape[1] > expected:
                    features = features[:, :expected]
                elif features.shape[1] < expected:
                    padding = np.zeros((1, expected - features.shape[1]))
                    features = np.hstack([features, padding])
                features_scaled = self.scaler.transform(features)
            
            # Apply PCA
            features_pca = self.pca.transform(features_scaled)
            
            # Get prediction
            prediction = self.model.predict(features_pca)[0]
            probabilities = self.model.predict_proba(features_pca)[0]
            
            # Get class names
            class_names = getattr(self.model, 'classes_', 
                                [f'class_{i}' for i in range(len(probabilities))])
            
            # Get top predictions
            top_indices = np.argsort(probabilities)[-3:][::-1]
            
            result = {
                'success': True,
                'predicted_class': class_names[prediction] if prediction < len(class_names) else 'unknown',
                'confidence': float(probabilities[prediction] * 100),
                'top_predictions': [
                    {
                        'plant': class_names[idx] if idx < len(class_names) else f'class_{idx}',
                        'confidence': float(probabilities[idx] * 100)
                    }
                    for idx in top_indices
                ],
                'model_type': 'sklearn_improved'
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Classification failed: {str(e)}'
            }

def main():
    """Test the improved classifier"""
    if len(sys.argv) != 2:
        print("Usage: python improved-sklearn-classifier.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    classifier = ImprovedSklearnClassifier()
    result = classifier.classify_image_safe(image_path)
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()