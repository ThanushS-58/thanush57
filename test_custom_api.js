// Test script to verify your custom model API integration
// Run with: node test_custom_api.js

import fs from 'fs';
import fetch from 'node-fetch';
import FormData from 'form-data';

async function testCustomModelAPI() {
  try {
    console.log('🧪 Testing Custom Model API Integration...\n');
    
    // Test 1: Check if custom model is enabled
    console.log('1️⃣ Checking model availability...');
    
    // Create a test image (simple colored square as base64)
    const testImageBase64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    const testImageBuffer = Buffer.from(testImageBase64, 'base64');
    
    // Save test image temporarily
    fs.writeFileSync('test_plant.png', testImageBuffer);
    
    // Test the API
    const formData = new FormData();
    formData.append('image', fs.createReadStream('test_plant.png'));
    formData.append('language', 'hi');
    
    console.log('2️⃣ Making API request to /api/identify...');
    
    const response = await fetch('http://localhost:5000/api/identify', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const result = await response.json();
    
    console.log('3️⃣ API Response:');
    console.log('✅ Plant Name:', result.plant?.name);
    console.log('✅ Hindi Name:', result.plant?.hindiName);
    console.log('✅ Confidence:', result.plant?.confidence + '%');
    console.log('✅ Model Type:', result.modelType || 'not specified');
    
    // Check if custom model was used
    if (result.modelType === 'custom_tensorflow') {
      console.log('\n🎉 SUCCESS: Your custom model is working!');
    } else if (result.analysis?.includes('Custom AI model')) {
      console.log('\n🎉 SUCCESS: Your custom model is working!');
    } else {
      console.log('\n⚠️  Using fallback model (OpenAI or database)');
      console.log('   To use custom model:');
      console.log('   1. Set USE_CUSTOM_MODEL=true in environment');
      console.log('   2. Ensure models/mediplant_classifier.h5 exists');
      console.log('   3. Install: pip install tensorflow pillow numpy');
    }
    
    // Cleanup
    fs.unlinkSync('test_plant.png');
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    
    if (error.message.includes('ECONNREFUSED')) {
      console.log('\n💡 Make sure your server is running:');
      console.log('   npm run dev');
    }
    
    // Cleanup on error
    try {
      fs.unlinkSync('test_plant.png');
    } catch {}
  }
}

// Test enhanced API endpoint too
async function testEnhancedAPI() {
  try {
    console.log('\n🔬 Testing Enhanced API endpoint...');
    
    const testImageBase64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    const testImageBuffer = Buffer.from(testImageBase64, 'base64');
    
    fs.writeFileSync('test_plant2.png', testImageBuffer);
    
    const formData = new FormData();
    formData.append('image', fs.createReadStream('test_plant2.png'));
    formData.append('language', 'hi');
    
    const response = await fetch('http://localhost:5000/api/plants/identify-enhanced', {
      method: 'POST',
      body: formData
    });
    
    if (response.ok) {
      const result = await response.json();
      console.log('✅ Enhanced API working');
      console.log('✅ Confidence:', result.confidence + '%');
      console.log('✅ Database Match:', result.databaseMatch ? 'Yes' : 'No');
    }
    
    fs.unlinkSync('test_plant2.png');
    
  } catch (error) {
    console.log('⚠️  Enhanced API test failed:', error.message);
    try {
      fs.unlinkSync('test_plant2.png');
    } catch {}
  }
}

// Run tests
async function runAllTests() {
  await testCustomModelAPI();
  await testEnhancedAPI();
  
  console.log('\n📋 Integration Summary:');
  console.log('• Your API endpoints: /api/identify, /api/plants/identify-enhanced');
  console.log('• Custom model files: models/mediplant_classifier.h5');
  console.log('• Enable with: USE_CUSTOM_MODEL=true');
  console.log('• Install deps: pip install tensorflow pillow numpy');
}

runAllTests();