// Integration Guide: How to use your trained 10-plant model
// Add this to your server/ai-services.ts

const TRAINED_PLANTS = {
  'turmeric': { hindi: 'हल्दी', threshold: 85 },
  'neem': { hindi: 'नीम', threshold: 88 },
  'ginger': { hindi: 'अदरक', threshold: 82 },
  'aloe_vera': { hindi: 'एलोवेरा', threshold: 90 },
  'tulsi': { hindi: 'तुलसी', threshold: 86 },
  'ashwagandha': { hindi: 'अश्वगंधा', threshold: 83 },
  'amla': { hindi: 'आंवला', threshold: 87 },
  'mint': { hindi: 'पुदीना', threshold: 85 },
  'fenugreek': { hindi: 'मेथी', threshold: 81 },
  'cinnamon': { hindi: 'दालचीनी', threshold: 91 }
};

// Enhanced classification with confidence checking
export async function classifyWithCustomModel(imageBase64) {
  // Your existing custom model logic...
  const result = await customClassifier.classifyImage(imageBase64);
  
  const plantName = result.predicted_class;
  const confidence = result.confidence * 100;
  const threshold = TRAINED_PLANTS[plantName]?.threshold || 80;
  
  if (confidence < threshold) {
    return {
      plant: null,
      message: "Plant not clearly identifiable",
      confidence: confidence,
      suggestions: result.top_predictions?.slice(0, 3)
    };
  }
  
  // Get plant info from your database
  const plantInfo = await db.select().from(plants)
    .where(eq(plants.name, plantName)).first();
  
  return {
    plant: {
      name: plantInfo.name,
      hindiName: TRAINED_PLANTS[plantName].hindi,
      scientificName: plantInfo.scientificName,
      confidence: confidence,
      uses: plantInfo.hindiUses || plantInfo.uses,
      preparation: plantInfo.hindiPreparation || plantInfo.preparation,
      // ... other plant data
    },
    modelType: "custom_trained",
    isHighConfidence: confidence >= threshold
  };
}