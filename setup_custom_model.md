# How to Integrate Your Trained AI Model with MediPlant API

## Step 1: After Training Your Model

Once you run `python train_plant_classifier.py`, you'll have these files:
```
models/
├── mediplant_classifier.h5          # Your trained model
├── mediplant_classifier_classes.txt # Plant class names
└── mediplant_classifier_final.h5    # Final model version
```

## Step 2: Install Runtime Dependencies

On your server environment, install Python dependencies:
```bash
pip install tensorflow==2.13.0 pillow numpy
```

## Step 3: Enable Your Custom Model

Set environment variable to use your trained model:
```bash
export USE_CUSTOM_MODEL=true
```

Or add to your `.env` file:
```
USE_CUSTOM_MODEL=true
```

## Step 4: Test Your Integration

Your existing API endpoints will now use your custom model:

### Test Plant Identification:
```bash
curl -X POST http://localhost:5000/api/identify \
  -F "image=@/path/to/plant_image.jpg" \
  -F "language=hi"
```

### Expected Response:
```json
{
  "plant": {
    "name": "Turmeric",
    "hindiName": "हल्दी",
    "scientificName": "Curcuma longa", 
    "confidence": 94,
    "medicinalUses": ["Anti-inflammatory", "Digestive aid"]
  },
  "analysis": "Custom AI model identified this as Turmeric with 94% confidence",
  "modelType": "custom_tensorflow"
}
```

## Step 5: Frontend Integration

No changes needed! Your existing frontend code will automatically use the custom model:

```javascript
// This code works exactly the same
const formData = new FormData();
formData.append('image', imageFile);

const response = await fetch('/api/identify', {
  method: 'POST', 
  body: formData
});

const result = await response.json();
// Now uses YOUR trained model instead of OpenAI
```

## Verification

Check server logs for this message when making API calls:
```
Using custom TensorFlow model for classification
```

If you see this, your custom model is working!

## Fallback Behavior

If custom model fails, API automatically falls back to:
1. OpenAI GPT-4o (if API key available)
2. Database lookup (demo mode)

Your app always works, even if custom model has issues.