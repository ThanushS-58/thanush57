#!/usr/bin/env python3
"""
MediPlant AI - Custom Plant Classification Model Training
This script trains a deep learning model for medicinal plant identification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 100  # Adjust based on your plant species count

def create_dataset(data_dir):
    """
    Create dataset from directory structure:
    data_dir/
    ├── class_1/
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── class_2/
    │   ├── image1.jpg
    │   └── image2.jpg
    """
    dataset = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    
    validation_dataset = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation", 
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    
    return dataset, validation_dataset

def create_model(num_classes):
    """
    Create a CNN model using transfer learning with EfficientNetB0
    """
    # Load pre-trained EfficientNetB0 (trained on ImageNet)
    base_model = keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Add custom classification layers
    model = keras.Sequential([
        # Data augmentation layers
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        
        # Preprocessing
        layers.Rescaling(1./255),
        
        # Base model
        base_model,
        
        # Custom layers
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model(data_dir, model_save_path):
    """
    Train the medicinal plant classification model
    """
    print("🌿 MediPlant AI Model Training Started...")
    
    # Create datasets
    train_dataset, val_dataset = create_dataset(data_dir)
    
    # Get class names
    class_names = train_dataset.class_names
    print(f"📋 Found {len(class_names)} plant classes:")
    for i, name in enumerate(class_names[:10]):  # Show first 10
        print(f"  {i}: {name}")
    
    # Create model
    model = create_model(len(class_names))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )
    
    # Print model summary
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False
        )
    ]
    
    # Train model
    print("🚀 Starting training...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning phase
    print("🔧 Fine-tuning with unfrozen layers...")
    base_model = model.layers[4]  # The EfficientNetB0 layer
    base_model.trainable = True
    
    # Lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001/10),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )
    
    # Continue training
    fine_tune_epochs = 10
    total_epochs = EPOCHS + fine_tune_epochs
    
    history_fine = model.fit(
        train_dataset,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(model_save_path.replace('.h5', '_final.h5'))
    
    # Save class names
    with open('models/mediplant_classifier_classes.txt', 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    print(f"✅ Model training completed!")
    print(f"📁 Model saved: {model_save_path}")
    print(f"📊 Final validation accuracy: {max(history_fine.history['val_accuracy']):.4f}")
    
    return model, history

def evaluate_model(model_path, test_data_dir):
    """
    Evaluate the trained model on test data
    """
    # Load model
    model = keras.models.load_model(model_path)
    
    # Load test dataset
    test_dataset = keras.utils.image_dataset_from_directory(
        test_data_dir,
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    
    # Evaluate
    results = model.evaluate(test_dataset, verbose=0)
    print(f"📊 Test Results:")
    print(f"  Loss: {results[0]:.4f}")
    print(f"  Accuracy: {results[1]:.4f}")
    print(f"  Top-5 Accuracy: {results[2]:.4f}")
    
    return results

if __name__ == "__main__":
    # Paths - adjust these for your setup
    DATA_DIR = "medicinal_plants_dataset"  # Your dataset directory
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    MODEL_SAVE_PATH = "models/mediplant_classifier.h5"
    
    # Check if dataset exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Dataset directory not found: {DATA_DIR}")
        print("Please organize your dataset in the following structure:")
        print("medicinal_plants_dataset/")
        print("├── turmeric/")
        print("│   ├── image1.jpg")
        print("│   └── image2.jpg")
        print("├── neem/")
        print("│   ├── image1.jpg") 
        print("│   └── image2.jpg")
        print("└── ...")
        exit(1)
    
    # Train model
    model, history = train_model(DATA_DIR, MODEL_SAVE_PATH)
    
    print("🎉 Training completed successfully!")
    print("Next steps:")
    print("1. Test your model with new images")
    print("2. Integrate into your MediPlant AI application")
    print("3. Deploy to production")