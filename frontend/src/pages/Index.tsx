import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Upload, Brain, AlertTriangle, CheckCircle, Camera, Zap, ArrowLeft } from "lucide-react";
import { toast } from "@/hooks/use-toast";
import { sheepAnalysisApi, SheepAnalysisResult } from "@/services/sheepAnalysisApi";

const Index = () => {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<SheepAnalysisResult | null>(null);
  const [currentView, setCurrentView] = useState<"upload" | "results">("upload");
  const [progress, setProgress] = useState(0);

  // Dynamic progress bar effect
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isAnalyzing) {
      setProgress(0);
      interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 95) return prev;
          const increment = Math.random() * 15 + 5; // Random increment between 5-20
          return Math.min(prev + increment, 95);
        });
      }, 800);
    } else {
      setProgress(0);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isAnalyzing]);

  // Function to translate raw prediction to user-friendly text
  const translatePrediction = (rawPrediction: string): string => {
    switch (rawPrediction) {
      case "corpus_sheep_face_pain":
        return "Pain Detected";
      case "corpus_sheep_face_no_pain":
        return "Healthy";
      default:
        return rawPrediction; // fallback to raw value if unknown
    }
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string);
        setAnalysisResult(null);
        setCurrentView("upload");
      };
      reader.readAsDataURL(file);
    }
  };

  const analyzeImage = async () => {
    if (!uploadedFile) return;
    
    setIsAnalyzing(true);
    setCurrentView("results");
    
    try {
      console.log('Sending image to backend for analysis...');
      const result = await sheepAnalysisApi.analyzeSheepImage(uploadedFile);
      console.log('Analysis result:', result);
      
      // Complete the progress bar
      setProgress(100);
      
      // Wait a moment before showing results
      setTimeout(() => {
        setAnalysisResult(result);
        
        const isPain = result.pain_probability > 0.5;
        toast({
          title: "Analysis Complete",
          description: `Sheep ${isPain ? "shows signs of discomfort" : "appears healthy"}`,
        });
      }, 500);
    } catch (error) {
      console.error('Analysis failed:', error);
      toast({
        title: "Analysis Failed",
        description: "Unable to connect to the backend. Please check your connection.",
        variant: "destructive",
      });
      setCurrentView("upload");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetToUpload = () => {
    setCurrentView("upload");
    setAnalysisResult(null);
    setIsAnalyzing(false);
    setProgress(0);
  };

  const isPainDetected = analysisResult ? analysisResult.pain_probability > 0.5 : false;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Mobile-first header */}
      <header className="sticky top-0 z-50 bg-black/20 backdrop-blur-xl border-b border-purple-500/20">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            {currentView === "results" && (
              <Button
                variant="ghost"
                size="sm"
                onClick={resetToUpload}
                className="text-purple-300 hover:text-white hover:bg-purple-500/20"
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back
              </Button>
            )}
            
            <div className="flex items-center justify-center space-x-3 flex-1">
              <div className="relative">
                <div className="w-10 h-10 bg-gradient-to-br from-purple-400 to-pink-400 rounded-xl flex items-center justify-center shadow-lg">
                  <span className="text-xl font-bold text-white">🐑</span>
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
              </div>
              <div className="text-center">
                <h1 className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                  FaceFarm
                </h1>
                <p className="text-xs text-purple-300">AI Sheep Health Monitor</p>
              </div>
            </div>

            {currentView === "results" && <div className="w-16"></div>}
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="container mx-auto px-4 py-8">
        {currentView === "upload" ? (
          /* Upload View */
          <Card className="bg-black/40 backdrop-blur-xl border-purple-500/30 shadow-2xl">
            <CardContent className="p-6">
              <div className="text-center mb-6">
                <h2 className="text-2xl font-bold text-white mb-2">Sheep Health Analysis</h2>
                <p className="text-purple-300">Upload a sheep image to detect signs of pain or discomfort</p>
              </div>

              <div className="space-y-6">
                {/* Image upload area */}
                <div className="relative">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    className="hidden"
                    id="image-upload"
                  />
                  <label 
                    htmlFor="image-upload" 
                    className="block cursor-pointer group"
                  >
                    <div className="border-2 border-dashed border-purple-500/50 rounded-2xl p-8 text-center hover:border-purple-400 transition-all duration-300 group-hover:bg-purple-500/5">
                      {uploadedImage ? (
                        <div className="space-y-4">
                          <div className="relative mx-auto w-full max-w-sm">
                            <img
                              src={uploadedImage}
                              alt="Uploaded sheep"
                              className="w-full h-64 object-cover rounded-xl shadow-2xl"
                            />
                            <div className="absolute inset-0 rounded-xl bg-gradient-to-t from-black/50 to-transparent"></div>
                            <Camera className="absolute bottom-4 right-4 h-6 w-6 text-white" />
                          </div>
                          <p className="text-sm text-purple-300">Tap to change image</p>
                        </div>
                      ) : (
                        <div className="space-y-6">
                          <div className="relative mx-auto w-20 h-20">
                            <Upload className="h-20 w-20 text-purple-400 mx-auto" />
                            <div className="absolute -bottom-2 -right-2 w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center">
                              <Zap className="h-4 w-4 text-white" />
                            </div>
                          </div>
                          <div>
                            <p className="text-xl font-semibold text-white mb-2">Drop your image here</p>
                            <p className="text-purple-300">or tap to browse</p>
                          </div>
                        </div>
                      )}
                    </div>
                  </label>
                </div>

                {/* Analyze button */}
                {uploadedImage && (
                  <Button
                    onClick={analyzeImage}
                    className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-semibold py-4 rounded-xl shadow-lg transition-all duration-300 transform hover:scale-105"
                    size="lg"
                  >
                    <Brain className="h-5 w-5 mr-3" />
                    Analyze Sheep Health
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>
        ) : (
          /* Results View */
          <Card className="bg-black/40 backdrop-blur-xl border-purple-500/30 shadow-2xl">
            <CardContent className="p-6">
              <div className="text-center mb-6">
                <h3 className="text-xl font-bold text-white flex items-center justify-center space-x-2">
                  <Brain className="h-5 w-5 text-purple-400" />
                  <span>Health Analysis Results</span>
                </h3>
              </div>

              {isAnalyzing ? (
                <div className="space-y-6">
                  <div className="text-center">
                    <div className="relative mx-auto w-20 h-20 mb-6">
                      <Brain className="h-20 w-20 text-purple-400 animate-pulse" />
                      <div className="absolute inset-0 rounded-full border-2 border-purple-400 border-t-transparent animate-spin"></div>
                    </div>
                    <p className="text-purple-300 font-medium text-lg mb-2">AI is analyzing sheep facial expressions...</p>
                    <p className="text-purple-400 text-sm">Processing health indicators and pain patterns</p>
                  </div>
                  
                  <div className="space-y-4">
                    <div className="flex justify-between text-sm">
                      <span className="text-purple-300">Analysis Progress</span>
                      <span className="text-purple-300 font-medium">{Math.round(progress)}%</span>
                    </div>
                    <Progress 
                      value={progress} 
                      className="w-full h-3 bg-purple-900/50"
                    />
                  </div>
                  
                  <div className="bg-gradient-to-r from-purple-900/30 to-pink-900/30 rounded-xl p-4 text-center border border-purple-500/20">
                    <p className="text-sm text-purple-300 mb-2">
                      Analyzing facial action units and pain indicators
                    </p>
                    <div className="flex justify-center space-x-2">
                      <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                  </div>
                </div>
              ) : analysisResult ? (
                <div className="space-y-6">
                  <div className="text-center">
                    {isPainDetected ? (
                      <div className="space-y-4">
                        <div className="relative mx-auto w-20 h-20">
                          <AlertTriangle className="h-20 w-20 text-red-400 mx-auto" />
                          <div className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 rounded-full animate-pulse"></div>
                        </div>
                        <div className="space-y-2">
                          <Badge className="bg-gradient-to-r from-red-600 to-red-500 text-white text-lg px-6 py-2 rounded-full">
                            {translatePrediction(analysisResult.prediction)}
                          </Badge>
                          <p className="text-red-300 text-sm">
                            The sheep shows facial indicators consistent with discomfort or pain
                          </p>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <div className="relative mx-auto w-20 h-20">
                          <CheckCircle className="h-20 w-20 text-green-400 mx-auto" />
                          <div className="absolute -top-2 -right-2 w-6 h-6 bg-green-500 rounded-full animate-pulse"></div>
                        </div>
                        <div className="space-y-2">
                          <Badge className="bg-gradient-to-r from-green-600 to-green-500 text-white text-lg px-6 py-2 rounded-full">
                            {translatePrediction(analysisResult.prediction)}
                          </Badge>
                          <p className="text-green-300 text-sm">
                            The sheep displays normal facial expressions with no signs of distress
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-white font-medium">Pain Probability</span>
                      <span className="text-purple-300 font-bold">
                        {(analysisResult.pain_probability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress 
                      value={analysisResult.pain_probability * 100} 
                      className="w-full h-3 bg-purple-900/50"
                    />
                  </div>

                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-white font-medium">Confidence Level</span>
                      <span className="text-purple-300 font-bold">
                        {(analysisResult.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress 
                      value={analysisResult.confidence * 100} 
                      className="w-full h-3 bg-purple-900/50"
                    />
                  </div>
                  
                  <div className="bg-gradient-to-r from-purple-900/50 to-pink-900/50 rounded-xl p-4 space-y-3">
                    <h4 className="font-bold text-white text-lg">Analysis Details</h4>
                    <div className="space-y-2 text-sm">
                      <p className="text-purple-300">
                        <span className="text-white font-medium">Filename:</span> {analysisResult.filename}
                      </p>
                      <p className="text-purple-300">
                        <span className="text-white font-medium">Model:</span> FaceFarm-Pain-V2.1
                      </p>
                      {isPainDetected && (
                        <div className="mt-4 p-3 bg-red-900/50 rounded-lg border border-red-500/30">
                          <p className="text-red-300 font-semibold flex items-center space-x-2">
                            <AlertTriangle className="h-4 w-4" />
                            <span>Veterinary consultation recommended</span>
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ) : null}
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  );
};

export default Index;
