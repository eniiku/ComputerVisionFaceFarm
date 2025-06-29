import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Upload,
  Brain,
  AlertTriangle,
  CheckCircle,
  Camera,
  ArrowLeft,
  History,
  X,
} from "lucide-react";
import { toast } from "@/hooks/use-toast";
import SheepRecordsHistory from "@/components/SheepRecordsHistory";
import { SheepAnalysisResponse } from "@/types/sheepRecord";
import { sheepAnalysisApi } from "@/services/sheepAnalysisApi";

const Index = () => {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] =
    useState<SheepAnalysisResponse | null>(null);
  const [currentView, setCurrentView] = useState<
    "upload" | "results" | "history" | "camera"
  >("upload");
  const [progress, setProgress] = useState(0);
  const [showCamera, setShowCamera] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

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

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment", // Use back camera on mobile
        },
      });
      setStream(mediaStream);
      setShowCamera(true);
      setCurrentView("camera");

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (error) {
      console.error("Error accessing camera:", error);
      toast({
        title: "Camera Error",
        description:
          "Unable to access camera. Please check permissions or try uploading an image instead.",
        variant: "destructive",
      });
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
    setShowCamera(false);
    setCurrentView("upload");
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext("2d");

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      if (context) {
        context.drawImage(video, 0, 0);

        canvas.toBlob(
          (blob) => {
            if (blob) {
              const file = new File([blob], `sheep-photo-${Date.now()}.jpg`, {
                type: "image/jpeg",
              });
              setUploadedFile(file);

              const reader = new FileReader();
              reader.onload = (e) => {
                setUploadedImage(e.target?.result as string);
                setAnalysisResult(null);
                stopCamera();
              };
              reader.readAsDataURL(file);
            }
          },
          "image/jpeg",
          0.8
        );
      }
    }
  };

  const analyzeImage = async () => {
    if (!uploadedFile) return;

    setIsAnalyzing(true);
    setCurrentView("results");
    try {
      console.log("Sending image to backend for analysis...");
      const result = await sheepAnalysisApi.analyzeSheepImage(uploadedFile);
      console.log("Analysis result:", result);
      setProgress(100);

      setTimeout(() => {
        setAnalysisResult(result);

        console.log(
          "Record processing (saving) handled by backend. Record ID:",
          result.record_id
        );

        const isPain = result.prediction === "pain";
        toast({
          title: "Analysis Complete",
          description: `Sheep ${
            isPain ? "shows signs of discomfort" : "appears healthy"
          }`,
          variant: isPain ? "destructive" : "default",
        });
      }, 500);
    } catch (error) {
      console.error("Analysis failed:", error);
      toast({
        title: "Analysis Failed",
        description:
          "Unable to connect to the backend or prediction failed. Please check your connection and backend logs.",
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

  const showHistory = () => {
    setCurrentView("history");
  };

  const isPainDetected = analysisResult
    ? analysisResult.pain_probability > 0.5
    : false;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Mobile-first header */}
      <header className="sticky top-0 z-50 bg-black/20 backdrop-blur-xl border-b border-purple-500/20">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            {(currentView === "results" || currentView === "history") && (
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
                <p className="text-xs text-purple-300">
                  AI Sheep Health Monitor
                </p>
              </div>
            </div>

            {currentView === "upload" && (
              <Button
                variant="ghost"
                size="sm"
                onClick={showHistory}
                className="text-purple-300 hover:text-white hover:bg-purple-500/20"
              >
                <History className="h-4 w-4 mr-2" />
                Records
              </Button>
            )}

            {(currentView === "results" || currentView === "history") && (
              <div className="w-16"></div>
            )}
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="container mx-auto px-4 py-8">
        {currentView === "history" ? (
          <SheepRecordsHistory onBack={resetToUpload} />
        ) : currentView === "camera" ? (
          // Camera View
          <Card className="bg-black/40 backdrop-blur-xl border-purple-500/30 shadow-2xl">
            <CardContent className="p-6">
              <div className="text-center mb-6">
                <h2 className="text-2xl font-bold text-white mb-2">
                  Take Photo
                </h2>
                <p className="text-purple-300">
                  Position the sheep in the camera view and tap capture
                </p>
              </div>

              <div className="space-y-6">
                <div className="relative">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    className="w-full h-64 object-cover rounded-xl bg-black"
                  />
                  <canvas ref={canvasRef} className="hidden" />
                </div>

                <div className="flex space-x-4">
                  <Button
                    onClick={capturePhoto}
                    className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-semibold py-4 rounded-xl shadow-lg transition-all duration-300"
                    size="lg"
                  >
                    <Camera className="h-5 w-5 mr-3" />
                    Capture Photo
                  </Button>
                  <Button
                    onClick={stopCamera}
                    variant="outline"
                    className="px-6 py-4 border-purple-500/50 text-purple-300 hover:bg-purple-500/20"
                  >
                    <X className="h-5 w-5" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ) : currentView === "upload" ? (
          // Upload View
          <Card className="bg-black/40 backdrop-blur-xl border-purple-500/30 shadow-2xl">
            <CardContent className="p-6">
              <div className="text-center mb-6">
                <h2 className="text-2xl font-bold text-white mb-2">
                  Sheep Health Analysis
                </h2>
                <p className="text-purple-300">
                  Upload a sheep image or take a photo to detect signs of pain
                  or discomfort
                </p>
              </div>

              <div className="space-y-6">
                {/* Camera and Upload buttons */}
                {!uploadedImage && (
                  <div className="grid grid-cols-2 gap-4">
                    <Button
                      onClick={startCamera}
                      className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white font-semibold py-4 rounded-xl shadow-lg transition-all duration-300"
                      size="lg"
                    >
                      <Camera className="h-5 w-5 mr-3" />
                      Take Photo
                    </Button>
                    <div>
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleImageUpload}
                        className="hidden"
                        id="image-upload"
                      />
                      <label htmlFor="image-upload">
                        <Button
                          asChild
                          className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-semibold py-4 rounded-xl shadow-lg transition-all duration-300"
                          size="lg"
                        >
                          <span className="cursor-pointer flex items-center justify-center">
                            <Upload className="h-5 w-5 mr-3" />
                            Upload Image
                          </span>
                        </Button>
                      </label>
                    </div>
                  </div>
                )}

                {/* Image display area */}
                {uploadedImage && (
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
                    <div className="flex space-x-2">
                      <Button
                        onClick={startCamera}
                        variant="outline"
                        className="flex-1 border-purple-500/50 text-purple-300 hover:bg-purple-500/20"
                      >
                        <Camera className="h-4 w-4 mr-2" />
                        Take New Photo
                      </Button>
                      <div className="flex-1">
                        <input
                          type="file"
                          accept="image/*"
                          onChange={handleImageUpload}
                          className="hidden"
                          id="image-upload-2"
                        />
                        <label htmlFor="image-upload-2" className="block">
                          <Button
                            asChild
                            variant="outline"
                            className="w-full border-purple-500/50 text-purple-300 hover:bg-purple-500/20"
                          >
                            <span className="cursor-pointer flex items-center justify-center">
                              <Upload className="h-4 w-4 mr-2" />
                              Upload Different
                            </span>
                          </Button>
                        </label>
                      </div>
                    </div>
                  </div>
                )}

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
          // Results View
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
                    <p className="text-purple-300 font-medium text-lg mb-2">
                      AI is analyzing sheep facial expressions...
                    </p>
                    <p className="text-purple-400 text-sm">
                      Processing health indicators and pain patterns
                    </p>
                  </div>

                  <div className="space-y-4">
                    <div className="flex justify-between text-sm">
                      <span className="text-purple-300">Analysis Progress</span>
                      <span className="text-purple-300 font-medium">
                        {Math.round(progress)}%
                      </span>
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
                      <div
                        className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"
                        style={{ animationDelay: "0.1s" }}
                      ></div>
                      <div
                        className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"
                        style={{ animationDelay: "0.2s" }}
                      ></div>
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
                            The sheep shows facial indicators consistent with
                            discomfort or pain
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
                            The sheep displays normal facial expressions with no
                            signs of distress
                          </p>
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-white font-medium">
                        Pain Probability
                      </span>
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
                      <span className="text-white font-medium">
                        Confidence Level
                      </span>
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
                    <h4 className="font-bold text-white text-lg">
                      Analysis Details
                    </h4>
                    <div className="space-y-2 text-sm">
                      <p className="text-purple-300">
                        <span className="text-white font-medium">
                          Filename:
                        </span>{" "}
                        {analysisResult.filename}
                      </p>
                      <p className="text-purple-300">
                        <span className="text-white font-medium">Model:</span>{" "}
                        FaceFarm-Pain-V2.1
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
