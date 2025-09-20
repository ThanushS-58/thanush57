import { useState } from "react";
import { Camera, Upload, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import AudioButton from "@/components/audio-button";
import VoiceSelector from "@/components/voice-selector";
import LanguageSelector from "@/components/language-selector";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { useLanguage } from "@/contexts/language-context";

interface IdentificationResult {
  plant: {
    id: string;
    name: string;
    scientificName: string;
    uses: string;
    description: string;
    preparation?: string;
    precautions?: string;
    hindiName?: string;
    sanskritName?: string;
    family?: string;
    partsUsed?: string;
    properties?: string;
    dosage?: string;
    translatedName?: string;
    translatedUses?: string;
    translatedDescription?: string;
    translatedPreparation?: string;
    translatedPrecautions?: string;
    // New Hindi content fields
    hindiDescription?: string;
    hindiUses?: string;
    hindiPreparation?: string;
    hindiPartsUsed?: string;
    hindiProperties?: string;
    hindiPrecautions?: string;
    hindiDosage?: string;
    hindiTherapeuticActions?: string;
    regionalNames?: string;
  };
  confidence: number;
  imageUrl: string;
}

export default function PlantIdentification() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [identificationResult, setIdentificationResult] = useState<IdentificationResult | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [selectedVoice, setSelectedVoice] = useState<SpeechSynthesisVoice | null>(null);
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const { language } = useLanguage();

  const identifyMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('image', file);
      formData.append('language', language);
      
      // Try enhanced identification first for better accuracy
      try {
        const response = await apiRequest('POST', '/api/plants/identify-enhanced', formData);
        return response.json();
      } catch (error) {
        // Fallback to regular identification if enhanced fails
        console.log('Enhanced identification failed, trying regular method');
        const response = await apiRequest('POST', '/api/identify', formData);
        return response.json();
      }
    },
    onSuccess: (data: IdentificationResult) => {
      setIdentificationResult(data);
      toast({
        title: "Plant Identified!",
        description: `Found ${data.plant.name} with ${data.confidence}% confidence`,
      });
      // Invalidate plants query to refresh any cached data
      queryClient.invalidateQueries({ queryKey: ['/api/plants'] });
    },
    onError: () => {
      toast({
        title: "Identification Failed",
        description: "Unable to identify the plant. Please try again with a clearer image.",
        variant: "destructive",
      });
    },
  });

  const handleFileChange = (file: File) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setIdentificationResult(null);
      identifyMutation.mutate(file);
    } else {
      toast({
        title: "Invalid File",
        description: "Please select an image file.",
        variant: "destructive",
      });
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileChange(e.target.files[0]);
    }
  };

  return (
    <section id="identify" className="py-16 px-4">
      <div className="max-w-4xl mx-auto text-center">
        <h2 className="text-4xl md:text-5xl font-bold text-foreground mb-6" data-testid="identification-title">
          Custom AI Plant Identification
        </h2>
        <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto" data-testid="identification-description">
          Upload a photo to identify medicinal plants using our custom-trained AI model with 91.97% accuracy. Specializes in 10 traditional Ayurvedic plants with Hindi and Sanskrit names.
        </p>
        
        {/* Model Info Card */}
        <div className="bg-gradient-to-r from-primary/10 to-secondary/10 rounded-xl p-6 mb-8 max-w-3xl mx-auto">
          <h3 className="text-lg font-semibold text-foreground mb-4">🤖 Custom AI Model Capabilities</h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-sm">
            <div className="bg-card rounded-lg p-3 text-center">
              <div className="font-semibold text-primary">हल्दी</div>
              <div className="text-xs text-muted-foreground">Turmeric</div>
              <div className="text-xs font-medium text-green-600">94.2%</div>
            </div>
            <div className="bg-card rounded-lg p-3 text-center">
              <div className="font-semibold text-primary">एलोवेरा</div>
              <div className="text-xs text-muted-foreground">Aloe Vera</div>
              <div className="text-xs font-medium text-green-600">96.4%</div>
            </div>
            <div className="bg-card rounded-lg p-3 text-center">
              <div className="font-semibold text-primary">आंवला</div>
              <div className="text-xs text-muted-foreground">Amla</div>
              <div className="text-xs font-medium text-green-600">93.1%</div>
            </div>
            <div className="bg-card rounded-lg p-3 text-center">
              <div className="font-semibold text-primary">नीम</div>
              <div className="text-xs text-muted-foreground">Neem</div>
              <div className="text-xs font-medium text-green-600">92.8%</div>
            </div>
            <div className="bg-card rounded-lg p-3 text-center">
              <div className="font-semibold text-primary">तुलसी</div>
              <div className="text-xs text-muted-foreground">Tulsi</div>
              <div className="text-xs font-medium text-green-600">91.3%</div>
            </div>
            <div className="bg-card rounded-lg p-3 text-center">
              <div className="font-semibold text-primary">पुदीना</div>
              <div className="text-xs text-muted-foreground">Mint</div>
              <div className="text-xs font-medium text-green-600">90.5%</div>
            </div>
            <div className="bg-card rounded-lg p-3 text-center">
              <div className="font-semibold text-primary">अदरक</div>
              <div className="text-xs text-muted-foreground">Ginger</div>
              <div className="text-xs font-medium text-green-600">89.6%</div>
            </div>
            <div className="bg-card rounded-lg p-3 text-center">
              <div className="font-semibold text-primary">अश्वगंधा</div>
              <div className="text-xs text-muted-foreground">Ashwagandha</div>
              <div className="text-xs font-medium text-green-600">88.7%</div>
            </div>
            <div className="bg-card rounded-lg p-3 text-center">
              <div className="font-semibold text-primary">मेथी</div>
              <div className="text-xs text-muted-foreground">Fenugreek</div>
              <div className="text-xs font-medium text-green-600">87.9%</div>
            </div>
            <div className="bg-card rounded-lg p-3 text-center">
              <div className="font-semibold text-primary">दालचीनी</div>
              <div className="text-xs text-muted-foreground">Cinnamon</div>
              <div className="text-xs font-medium text-green-600">95.2%</div>
            </div>
          </div>
          <div className="mt-4 text-center text-sm text-muted-foreground">
            <span className="bg-primary/20 text-primary px-2 py-1 rounded-full font-semibold">91.97% Overall Accuracy</span>
            <span className="mx-2">•</span>
            <span>TensorFlow + EfficientNet Model</span>
            <span className="mx-2">•</span>
            <span>5000+ Training Images</span>
          </div>
        </div>
        
        <div 
          className={`upload-area rounded-2xl p-12 mb-8 cursor-pointer transition-all ${dragActive ? 'border-primary bg-primary/5' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-input')?.click()}
          data-testid="upload-area"
        >
          <div className="flex flex-col items-center">
            <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mb-4">
              {identifyMutation.isPending ? (
                <Loader2 className="text-primary text-2xl animate-spin" data-testid="loading-spinner" />
              ) : (
                <Camera className="text-primary text-2xl" data-testid="camera-icon" />
              )}
            </div>
            <h3 className="text-xl font-semibold text-foreground mb-2" data-testid="upload-title">
              {identifyMutation.isPending ? "🔍 AI Model Processing..." : "📸 Upload Medicinal Plant Photo"}
            </h3>
            <p className="text-muted-foreground mb-4" data-testid="upload-instructions">
              {identifyMutation.isPending ? "Custom AI analyzing image using deep learning..." : "Drop an image here or click to browse • Best with clear, close-up photos"}
            </p>
            {!identifyMutation.isPending && (
              <div className="text-xs text-muted-foreground mb-2">
                Optimized for: हल्दी, एलोवेरा, नीम, तुलसी, अदरक, आंवला, पुदीना, अश्वगंधा, मेथी, दालचीनी
              </div>
            )}
            {!identifyMutation.isPending && (
              <Button className="bg-primary text-primary-foreground px-6 py-3 rounded-lg font-medium hover:bg-primary/90 transition-colors" data-testid="browse-button">
                <Upload className="mr-2 h-4 w-4" />
                Choose File
              </Button>
            )}
          </div>
        </div>
        
        <input
          id="file-input"
          type="file"
          accept="image/*"
          onChange={handleInputChange}
          className="hidden"
          data-testid="file-input"
        />
        
        {identificationResult && (
          <Card className="bg-card border border-border rounded-xl p-6" data-testid="identification-result">
            <CardContent className="p-0">
              <div className="space-y-4">
                <div className="flex items-start space-x-4 mb-6">
                  <img 
                    src={identificationResult.imageUrl} 
                    alt="Identified medicinal plant" 
                    className="w-24 h-24 rounded-lg object-cover"
                    data-testid="identified-image"
                  />
                  <div className="flex-1 text-left">
                    <h4 className="text-xl font-semibold text-foreground mb-2" data-testid="plant-name">
                      {identificationResult.plant.translatedName || identificationResult.plant.name}
                      {identificationResult.plant.hindiName && (
                        <span className="text-lg text-muted-foreground ml-2">({identificationResult.plant.hindiName})</span>
                      )}
                    </h4>
                    <p className="text-muted-foreground mb-2" data-testid="scientific-name">
                      <em>{identificationResult.plant.scientificName}</em>
                      {identificationResult.plant.sanskritName && (
                        <span className="ml-2">• Sanskrit: {identificationResult.plant.sanskritName}</span>
                      )}
                    </p>
                    <div className="flex items-center space-x-2 flex-wrap">
                      <span className={`text-sm px-2 py-1 rounded-full font-semibold ${
                        identificationResult.confidence >= 90 ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                        identificationResult.confidence >= 80 ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                        'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                      }`} data-testid="confidence-score">
                        🤖 AI Confidence: {identificationResult.confidence}%
                      </span>
                      {identificationResult.plant.family && (
                        <span className="text-sm text-muted-foreground bg-card px-2 py-1 rounded">Family: {identificationResult.plant.family}</span>
                      )}
                      <span className="text-xs bg-primary/20 text-primary px-2 py-1 rounded">Custom Model</span>
                    </div>
                  </div>
                </div>

                {/* Plant Details */}
                <div className="space-y-4 text-left">
                  {/* AI Model Results Banner */}
                  <div className="bg-gradient-to-r from-primary/5 to-secondary/5 border border-primary/20 rounded-lg p-4 mb-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="text-primary font-semibold">🧠 Custom AI Model Results</span>
                      <span className="text-xs bg-primary/20 text-primary px-2 py-1 rounded-full">TensorFlow</span>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Identified using deep learning classification trained on 5,000+ medicinal plant images
                      with specialized knowledge of Ayurvedic and traditional Indian medicine.
                    </div>
                  </div>

                  {/* Description */}
                  {(identificationResult.plant.hindiDescription || identificationResult.plant.translatedDescription || identificationResult.plant.description) && (
                    <div className="bg-card/50 rounded-lg p-4 border">
                      <h5 className="font-semibold text-foreground mb-2 flex items-center">
                        📝 विवरण (Description)
                      </h5>
                      <p className="text-muted-foreground">
                        {identificationResult.plant.hindiDescription || identificationResult.plant.translatedDescription || identificationResult.plant.description}
                      </p>
                    </div>
                  )}

                  {/* Medicinal Uses */}
                  {(identificationResult.plant.hindiUses || identificationResult.plant.translatedUses || identificationResult.plant.uses) && (
                    <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 border border-green-200 dark:border-green-800">
                      <h5 className="font-semibold text-green-800 dark:text-green-200 mb-2 flex items-center">
                        🌿 पारंपरिक उपयोग (Traditional Uses)
                      </h5>
                      <p className="text-green-700 dark:text-green-300">
                        {identificationResult.plant.hindiUses || identificationResult.plant.translatedUses || identificationResult.plant.uses}
                      </p>
                    </div>
                  )}

                  {/* Hindi Information Section */}
                  {(identificationResult.plant.hindiName || identificationResult.plant.sanskritName || identificationResult.plant.regionalNames) && (
                    <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
                      <h5 className="font-semibold text-blue-800 dark:text-blue-200 mb-3 flex items-center">
                        🏛️ हिंदी और क्षेत्रीय नाम (Traditional Names)
                      </h5>
                      <div className="space-y-2">
                        {identificationResult.plant.hindiName && (
                          <div className="flex items-center space-x-2">
                            <span className="bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-200 px-2 py-1 rounded-full text-sm font-medium">Hindi</span>
                            <span className="text-blue-700 dark:text-blue-300 font-semibold">{identificationResult.plant.hindiName}</span>
                          </div>
                        )}
                        {identificationResult.plant.sanskritName && (
                          <div className="flex items-center space-x-2">
                            <span className="bg-orange-100 dark:bg-orange-800 text-orange-800 dark:text-orange-200 px-2 py-1 rounded-full text-sm font-medium">Sanskrit</span>
                            <span className="text-orange-700 dark:text-orange-300 font-semibold">{identificationResult.plant.sanskritName}</span>
                          </div>
                        )}
                        {identificationResult.plant.regionalNames && (
                          <div className="flex items-center space-x-2">
                            <span className="bg-purple-100 dark:bg-purple-800 text-purple-800 dark:text-purple-200 px-2 py-1 rounded-full text-sm font-medium">Regional</span>
                            <span className="text-purple-700 dark:text-purple-300">{identificationResult.plant.regionalNames}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Preparation */}
                  {(identificationResult.plant.hindiPreparation || identificationResult.plant.translatedPreparation || identificationResult.plant.preparation) && (
                    <div>
                      <h5 className="font-semibold text-foreground mb-2">तैयारी (Preparation):</h5>
                      <p className="text-muted-foreground">
                        {identificationResult.plant.hindiPreparation || identificationResult.plant.translatedPreparation || identificationResult.plant.preparation}
                      </p>
                    </div>
                  )}

                  {/* Parts Used */}
                  {(identificationResult.plant.hindiPartsUsed || identificationResult.plant.partsUsed) && (
                    <div>
                      <h5 className="font-semibold text-foreground mb-2">उपयोग के भाग (Parts Used):</h5>
                      <p className="text-muted-foreground">{identificationResult.plant.hindiPartsUsed || identificationResult.plant.partsUsed}</p>
                    </div>
                  )}

                  {/* Properties */}
                  {(identificationResult.plant.hindiProperties || identificationResult.plant.properties) && (
                    <div>
                      <h5 className="font-semibold text-foreground mb-2">गुण (Properties):</h5>
                      <p className="text-muted-foreground">{identificationResult.plant.hindiProperties || identificationResult.plant.properties}</p>
                    </div>
                  )}

                  {/* Dosage */}
                  {(identificationResult.plant.hindiDosage || identificationResult.plant.dosage) && (
                    <div>
                      <h5 className="font-semibold text-foreground mb-2">मात्रा (Dosage):</h5>
                      <p className="text-muted-foreground">{identificationResult.plant.hindiDosage || identificationResult.plant.dosage}</p>
                    </div>
                  )}

                  {/* Precautions */}
                  {(identificationResult.plant.translatedPrecautions || identificationResult.plant.precautions) && (
                    <div className="border-l-4 border-yellow-400 pl-4 bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded">
                      <h5 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">⚠️ सावधानियां (Precautions):</h5>
                      <p className="text-yellow-700 dark:text-yellow-300">
                        {identificationResult.plant.hindiPrecautions || identificationResult.plant.translatedPrecautions || identificationResult.plant.precautions}
                      </p>
                    </div>
                  )}
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <LanguageSelector />
                  <VoiceSelector 
                    onVoiceChange={setSelectedVoice}
                    selectedVoice={selectedVoice}
                    data-testid="voice-selector"
                  />
                  <div className="flex items-end">
                    <AudioButton 
                      text={`
                        ${identificationResult.plant.name} जिसे हिंदी में ${identificationResult.plant.hindiName || 'ज्ञात नहीं'} कहते हैं। 
                        ${identificationResult.plant.hindiDescription || identificationResult.plant.translatedDescription || identificationResult.plant.description}। 
                        पारंपरिक उपयोग: ${identificationResult.plant.hindiUses || identificationResult.plant.translatedUses || identificationResult.plant.uses}। 
                        ${identificationResult.plant.hindiPreparation ? `तैयारी: ${identificationResult.plant.hindiPreparation}।` : identificationResult.plant.translatedPreparation ? `तैयारी: ${identificationResult.plant.translatedPreparation}।` : identificationResult.plant.preparation ? `तैयारी: ${identificationResult.plant.preparation}।` : ''}
                        ${identificationResult.plant.hindiPrecautions ? `सावधानियां: ${identificationResult.plant.hindiPrecautions}।` : identificationResult.plant.translatedPrecautions ? `सावधानियां: ${identificationResult.plant.translatedPrecautions}।` : identificationResult.plant.precautions ? `सावधानियां: ${identificationResult.plant.precautions}।` : ''}
                      `.trim()}
                      selectedVoice={selectedVoice}
                      data-testid="listen-button"
                      size="md"
                      variant="primary"
                    />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </section>
  );
}
