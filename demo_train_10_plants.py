#!/usr/bin/env python3
"""
MediPlant AI - Demo Training for 10 Medicinal Plants
This demonstrates how to train a model for 10 specific plants
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
import os
from datetime import datetime

# Configuration for 10 plants demo
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10  # Reduced for demo
NUM_CLASSES = 10

# 10 Popular Medicinal Plants for Demo
PLANT_CLASSES = [
    "turmeric",     # हल्दी
    "neem",         # नीम
    "ginger",       # अदरक
    "aloe_vera",    # एलोवेरा
    "tulsi",        # तुलसी
    "ashwagandha",  # अश्वगंधा
    "amla",         # आंवला
    "mint",         # पुदीना
    "fenugreek",    # मेथी
    "cinnamon"      # दालचीनी
]

# Plant information database
PLANT_INFO = {
    "turmeric": {
        "hindi_name": "हल्दी",
        "scientific_name": "Curcuma longa",
        "family": "Zingiberaceae",
        "uses": "Anti-inflammatory, wound healing, digestive",
        "parts_used": "Rhizome",
        "confidence_indicators": ["yellow", "orange", "rhizome", "underground"]
    },
    "neem": {
        "hindi_name": "नीम",
        "scientific_name": "Azadirachta indica",
        "family": "Meliaceae",
        "uses": "Antibacterial, antifungal, skin disorders",
        "parts_used": "Leaves, bark, oil",
        "confidence_indicators": ["green", "compound_leaves", "bitter"]
    },
    "ginger": {
        "hindi_name": "अदरक",
        "scientific_name": "Zingiber officinale",
        "family": "Zingiberaceae",
        "uses": "Digestive, anti-nausea, anti-inflammatory",
        "parts_used": "Rhizome",
        "confidence_indicators": ["beige", "brown", "rhizome", "knobby"]
    },
    "aloe_vera": {
        "hindi_name": "एलोवेरा",
        "scientific_name": "Aloe barbadensis",
        "family": "Asphodelaceae",
        "uses": "Skin healing, burns, moisturizing",
        "parts_used": "Leaves, gel",
        "confidence_indicators": ["green", "succulent", "thick_leaves", "gel"]
    },
    "tulsi": {
        "hindi_name": "तुलसी",
        "scientific_name": "Ocimum sanctum",
        "family": "Lamiaceae",
        "uses": "Respiratory, immunity, stress relief",
        "parts_used": "Leaves",
        "confidence_indicators": ["green", "aromatic", "small_leaves", "sacred"]
    },
    "ashwagandha": {
        "hindi_name": "अश्वगंधा",
        "scientific_name": "Withania somnifera",
        "family": "Solanaceae",
        "uses": "Adaptogen, stress relief, energy",
        "parts_used": "Root",
        "confidence_indicators": ["root", "brown", "woody", "adaptogen"]
    },
    "amla": {
        "hindi_name": "आंवला",
        "scientific_name": "Emblica officinalis",
        "family": "Phyllanthaceae",
        "uses": "Vitamin C, hair care, immunity",
        "parts_used": "Fruit",
        "confidence_indicators": ["green", "round", "fruit", "sour"]
    },
    "mint": {
        "hindi_name": "पुदीना",
        "scientific_name": "Mentha",
        "family": "Lamiaceae",
        "uses": "Digestive, cooling, respiratory",
        "parts_used": "Leaves",
        "confidence_indicators": ["green", "aromatic", "serrated_leaves", "cooling"]
    },
    "fenugreek": {
        "hindi_name": "मेथी",
        "scientific_name": "Trigonella foenum-graecum",
        "family": "Fabaceae",
        "uses": "Diabetes, lactation, digestive",
        "parts_used": "Seeds, leaves",
        "confidence_indicators": ["seeds", "yellow", "bitter", "trifoliate"]
    },
    "cinnamon": {
        "hindi_name": "दालचीनी",
        "scientific_name": "Cinnamomum verum",
        "family": "Lauraceae",
        "uses": "Blood sugar, antimicrobial, warming",
        "parts_used": "Bark",
        "confidence_indicators": ["brown", "bark", "aromatic", "spice"]
    }
}

def create_synthetic_dataset():
    """
    Create synthetic training data for demonstration
    In real scenario, you would load actual images
    """
    print("🔄 Creating synthetic dataset for 10 medicinal plants...")
    
    # Simulate different images per class
    images_per_class = 50  # Reduced for demo
    
    # Create synthetic image data (normally you'd load real images)
    X = []
    y = []
    
    for class_idx, plant_name in enumerate(PLANT_CLASSES):
        print(f"  📁 Processing {plant_name} ({PLANT_INFO[plant_name]['hindi_name']})...")
        
        for img_idx in range(images_per_class):
            # Create synthetic image with some class-specific patterns
            synthetic_img = create_plant_pattern(class_idx, img_idx)
            X.append(synthetic_img)
            y.append(class_idx)
    
    X = np.array(X)
    y = keras.utils.to_categorical(y, NUM_CLASSES)
    
    print(f"✅ Dataset created: {X.shape[0]} images, {NUM_CLASSES} classes")
    return X, y

def create_plant_pattern(class_idx, img_idx):
    """
    Create synthetic image patterns for different plants
    """
    # Create base image
    img = np.random.rand(IMG_SIZE, IMG_SIZE, 3) * 0.3
    
    # Add class-specific patterns
    if class_idx == 0:  # turmeric - yellow/orange patterns
        img[:, :, 0] += 0.6  # Red channel
        img[:, :, 1] += 0.5  # Green channel
        img[:, :, 2] += 0.1  # Blue channel
    elif class_idx == 1:  # neem - green leaf patterns
        img[:, :, 1] += 0.7  # Strong green
        img[:, :, 0] += 0.2
    elif class_idx == 2:  # ginger - brown/beige patterns
        img[:, :, 0] += 0.5
        img[:, :, 1] += 0.4
        img[:, :, 2] += 0.3
    # Add more patterns for other plants...
    
    # Add some random variation
    noise = np.random.rand(IMG_SIZE, IMG_SIZE, 3) * 0.1
    img += noise
    
    # Normalize
    img = np.clip(img, 0, 1)
    
    return img

def create_demo_model():
    """
    Create a lightweight model for demonstration
    """
    print("🏗️ Building plant identification model...")
    
    model = keras.Sequential([
        # Input layer
        keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # Data augmentation
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        
        # Convolutional layers
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        
        # Dense layers
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def train_demo_model():
    """
    Train the demonstration model
    """
    print("🌿 MediPlant AI - Training Demo for 10 Plants")
    print("=" * 50)
    
    # Create dataset
    X, y = create_synthetic_dataset()
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"📊 Training set: {X_train.shape[0]} images")
    print(f"📊 Validation set: {X_val.shape[0]} images")
    
    # Create model
    model = create_demo_model()
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    print("\n🚀 Starting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/mediplant_10_demo.h5")
    
    # Save class names
    with open("models/demo_classes.txt", "w") as f:
        for plant in PLANT_CLASSES:
            f.write(f"{plant}\n")
    
    # Save plant info
    with open("models/plant_info.json", "w") as f:
        json.dump(PLANT_INFO, f, indent=2, ensure_ascii=False)
    
    return model, history

def test_model_predictions():
    """
    Test the trained model with sample predictions
    """
    print("\n🧪 Testing Model Predictions...")
    print("=" * 40)
    
    # Load model
    model = keras.models.load_model("models/mediplant_10_demo.h5")
    
    # Test with synthetic images
    for i, plant_name in enumerate(PLANT_CLASSES[:5]):  # Test first 5 plants
        # Create test image
        test_img = create_plant_pattern(i, 0)
        test_img = np.expand_dims(test_img, axis=0)
        
        # Predict
        predictions = model.predict(test_img, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100
        
        plant_info = PLANT_INFO[plant_name]
        
        print(f"\n🌱 Plant: {plant_name.title()} ({plant_info['hindi_name']})")
        print(f"   Scientific: {plant_info['scientific_name']}")
        print(f"   Predicted: {PLANT_CLASSES[predicted_class]} ({confidence:.1f}% confidence)")
        print(f"   Uses: {plant_info['uses']}")
        print(f"   Parts Used: {plant_info['parts_used']}")
        
        if predicted_class == i:
            print("   ✅ CORRECT PREDICTION!")
        else:
            print("   ❌ Incorrect prediction")

def show_training_summary():
    """
    Show summary of what was accomplished
    """
    print("\n🎉 Training Demo Complete!")
    print("=" * 50)
    print("📋 What was trained:")
    for i, plant in enumerate(PLANT_CLASSES):
        info = PLANT_INFO[plant]
        print(f"  {i+1:2d}. {plant.title()} ({info['hindi_name']}) - {info['scientific_name']}")
    
    print(f"\n📁 Files created:")
    print(f"  • models/mediplant_10_demo.h5 - Trained model")
    print(f"  • models/demo_classes.txt - Plant class names") 
    print(f"  • models/plant_info.json - Plant information database")
    
    print(f"\n🎯 Model Performance:")
    print(f"  • Architecture: CNN with data augmentation")
    print(f"  • Classes: {NUM_CLASSES} medicinal plants")
    print(f"  • Training: {EPOCHS} epochs")
    print(f"  • Ready for integration with your MediPlant AI app!")

if __name__ == "__main__":
    # Run the complete demo
    model, history = train_demo_model()
    test_model_predictions()
    show_training_summary()
    
    print(f"\n🚀 Next Steps:")
    print(f"  1. Set USE_CUSTOM_MODEL=true in your environment")
    print(f"  2. Update your inference script to use this model")
    print(f"  3. Test with real plant images!")
    print(f"  4. Collect real image datasets for better accuracy")