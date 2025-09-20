#!/usr/bin/env python3
"""
MediPlant AI - Model Inference Script
This script performs inference on trained models for plant classification
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import json
import sys
import argparse
import os

def load_model_and_classes(model_path, classes_path):
    """Load the trained model and class names"""
    try:
        model = tf.keras.models.load_model(model_path)
        
        with open(classes_path, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines()]
        
        return model, class_names
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

def preprocess_image(image_path, img_size=128):
    """Preprocess image for model input - adjusted for your model's expected input size"""
    try:
        # Load and resize image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size (128x128 for your model)
        img = img.resize((img_size, img_size))
        
        # Convert to array and normalize
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize to [0,1]
        
        return img_array
    except Exception as e:
        raise Exception(f"Failed to preprocess image: {str(e)}")

def predict_plant(image_path, model_path, classes_path, top_k=5):
    """
    Predict plant species from image
    """
    try:
        # Load model and classes
        model, class_names = load_model_and_classes(model_path, classes_path)
        
        # Preprocess image
        img_array = preprocess_image(image_path)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_probabilities = predictions[0]
        
        # Get top prediction
        predicted_class_idx = np.argmax(predicted_probabilities)
        predicted_class = class_names[predicted_class_idx]
        confidence = float(predicted_probabilities[predicted_class_idx])
        
        # Get top-k predictions
        top_k_indices = np.argsort(predicted_probabilities)[-top_k:][::-1]
        top_k_predictions = []
        
        for idx in top_k_indices:
            top_k_predictions.append({
                'class': class_names[idx],
                'confidence': float(predicted_probabilities[idx]),
                'index': int(idx)
            })
        
        # Prepare result
        result = {
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top_predictions': top_k_predictions,
            'total_classes': len(class_names),
            'model_info': {
                'input_shape': model.input_shape,
                'output_shape': model.output_shape
            }
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'predicted_class': None,
            'confidence': 0.0,
            'top_predictions': []
        }

def main():
    parser = argparse.ArgumentParser(description='MediPlant AI Inference')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--classes', required=True, help='Path to classes file')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top predictions')
    parser.add_argument('--output', help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image):
        print(json.dumps({'success': False, 'error': f'Image file not found: {args.image}'}))
        sys.exit(1)
        
    if not os.path.exists(args.model):
        print(json.dumps({'success': False, 'error': f'Model file not found: {args.model}'}))
        sys.exit(1)
        
    if not os.path.exists(args.classes):
        print(json.dumps({'success': False, 'error': f'Classes file not found: {args.classes}'}))
        sys.exit(1)
    
    # Run prediction
    result = predict_plant(args.image, args.model, args.classes, args.top_k)
    
    # Output result
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    else:
        print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()