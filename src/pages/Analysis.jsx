import React, { useState, useEffect, useRef, useCallback } from "react";
import { 
  FileQuestion, 
  Cpu, 
  AlertTriangle, 
  Activity,
  ShieldCheck,
  BrainCircuit,
  Image as ImageIcon,
  Binary
} from "lucide-react";
import { apiService } from "../services/api";

export default function Analysis({ activeEvidence }) {
  const [analysisData, setAnalysisData] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingError, setProcessingError] = useState("");
  const [showFaceBoxes, setShowFaceBoxes] = useState(true);
  
  // Phase 4 Deepfake analysis states
  const [deepfakeData, setDeepfakeData] = useState(null);
  const [isAnalyzingDeepfake, setIsAnalyzingDeepfake] = useState(false);
  const [deepfakeError, setDeepfakeError] = useState("");

  // Phase 5 ELA & FFT signal analysis states
  const [elaData, setElaData] = useState(null);
  const [isAnalyzingEla, setIsAnalyzingEla] = useState(false);
  const [elaError, setElaError] = useState("");

  const [fftData, setFftData] = useState(null);
  const [isAnalyzingFft, setIsAnalyzingFft] = useState(false);
  const [fftError, setFftError] = useState("");

  const [activeSignalTab, setActiveSignalTab] = useState(() => {
    if (typeof window !== "undefined") {
      const tab = new URLSearchParams(window.location.search).get("signal_tab");
      if (tab === "fft" || tab === "ela") return tab;
    }
    return "ela";
  });

  // Phase 6 Grad-CAM Explainability states
  const [gradCamMap, setGradCamMap] = useState({});
  const [isGeneratingGradCam, setIsGeneratingGradCam] = useState(null);
  const [gradCamError, setGradCamError] = useState("");

  // Image sizing states for coordinate scaling
  const [naturalWidth, setNaturalWidth] = useState(0);
  const [naturalHeight, setNaturalHeight] = useState(0);
  const imgRef = useRef(null);

  // Video playback states
  const videoRef = useRef(null);
  const [selectedFrame, setSelectedFrame] = useState(null); // The frame index currently inspected

  // Load latest preprocessing results
  const fetchPreprocessingResults = useCallback(async () => {
    if (!activeEvidence) {
      setAnalysisData(null);
      setProcessingError("");
      return;
    }
    
    try {
      const data = await apiService.getPreprocessing(activeEvidence.id);
      setAnalysisData(data);
      setProcessingError("");
      
      // Auto-select first sampled frame with faces in video
      if (data && data.video_metadata && data.faces && data.faces.length > 0) {
        const uniqueFrames = Array.from(new Set(data.faces.map(f => f.frame_index))).sort((a,b) => a-b);
        if (uniqueFrames.length > 0) {
          setSelectedFrame(prev => (prev === null ? uniqueFrames[0] : prev));
        }
      }
    } catch (err) {
      console.error("Failed to load preprocessing results", err);
      setAnalysisData(null);
    }
  }, [activeEvidence]);

  // Load latest deepfake results
  const fetchDeepfakeResults = useCallback(async () => {
    if (!activeEvidence) {
      setDeepfakeData(null);
      setDeepfakeError("");
      return;
    }

    try {
      const data = await apiService.getDeepfakeResults(activeEvidence.id);
      setDeepfakeData(data);
      setDeepfakeError("");
    } catch {
      setDeepfakeData(null);
    }
  }, [activeEvidence]);

  // Load latest ELA results
  const fetchElaResults = useCallback(async () => {
    if (!activeEvidence || activeEvidence.media_category !== "IMAGE") {
      setElaData(null);
      setElaError("");
      return;
    }

    try {
      const data = await apiService.getElaResults(activeEvidence.id);
      setElaData(data);
      setElaError("");
    } catch {
      setElaData(null);
    }
  }, [activeEvidence]);

  // Load latest FFT results
  const fetchFftResults = useCallback(async () => {
    if (!activeEvidence || activeEvidence.media_category !== "IMAGE") {
      setFftData(null);
      setFftError("");
      return;
    }

    try {
      const data = await apiService.getFftResults(activeEvidence.id);
      setFftData(data);
      setFftError("");
    } catch {
      setFftData(null);
    }
  }, [activeEvidence]);

  useEffect(() => {
    setGradCamMap({});
    setGradCamError("");
    fetchPreprocessingResults();
    fetchDeepfakeResults();
    fetchElaResults();
    fetchFftResults();
  }, [activeEvidence, fetchPreprocessingResults, fetchDeepfakeResults, fetchElaResults, fetchFftResults]);

  const handleGenerateGradCam = useCallback(async (faceId) => {
    if (!activeEvidence) return;
    setIsGeneratingGradCam(faceId);
    setGradCamError("");
    try {
      const res = await apiService.generateGradCam(activeEvidence.id, faceId);
      setGradCamMap(prev => ({ ...prev, [faceId]: res.gradcam_filename }));
    } catch (err) {
      setGradCamError(err.message || "Failed to generate Grad-CAM visualization.");
    } finally {
      setIsGeneratingGradCam(null);
    }
  }, [activeEvidence]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      if (params.get("auto_gradcam") === "true" && analysisData?.faces?.length > 0 && Object.keys(gradCamMap).length === 0 && isGeneratingGradCam === null) {
        handleGenerateGradCam(analysisData.faces[0].id);
      }
    }
  }, [analysisData, gradCamMap, isGeneratingGradCam, handleGenerateGradCam]);

  const handleStartPreprocessing = async () => {
    if (!activeEvidence) return;
    
    setIsProcessing(true);
    setProcessingError("");
    try {
      await apiService.startPreprocessing(activeEvidence.id);
      await fetchPreprocessingResults();
      setDeepfakeData(null);
    } catch (err) {
      setProcessingError(err.message || "An unexpected error occurred during preprocessing.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleStartDeepfakeAnalysis = async () => {
    if (!activeEvidence) return;

    setIsAnalyzingDeepfake(true);
    setDeepfakeError("");
    try {
      await apiService.startDeepfakeAnalysis(activeEvidence.id);
      await fetchDeepfakeResults();
    } catch (err) {
      setDeepfakeError(err.message || "Deepfake analysis pipeline run failed.");
    } finally {
      setIsAnalyzingDeepfake(false);
    }
  };

  const handleStartElaAnalysis = async () => {
    if (!activeEvidence) return;

    setIsAnalyzingEla(true);
    setElaError("");
    try {
      await apiService.startElaAnalysis(activeEvidence.id);
      await fetchElaResults();
    } catch (err) {
      setElaError(err.message || "Error Level Analysis failed.");
    } finally {
      setIsAnalyzingEla(false);
    }
  };

  const handleStartFftAnalysis = async () => {
    if (!activeEvidence) return;

    setIsAnalyzingFft(true);
    setFftError("");
    try {
      await apiService.startFftAnalysis(activeEvidence.id);
      await fetchFftResults();
    } catch (err) {
      setFftError(err.message || "Fast Fourier Transform failed.");
    } finally {
      setIsAnalyzingFft(false);
    }
  };

  const handleImageLoad = (e) => {
    setNaturalWidth(e.target.naturalWidth);
    setNaturalHeight(e.target.naturalHeight);
  };

  const getActiveFaces = () => {
    if (!analysisData || !analysisData.faces) return [];
    if (activeEvidence.media_category === "IMAGE") {
      return analysisData.faces;
    }
    return analysisData.faces.filter(f => f.frame_index === selectedFrame);
  };

  useEffect(() => {
    if (videoRef.current && selectedFrame !== null && analysisData?.faces) {
      const face = analysisData.faces.find(f => f.frame_index === selectedFrame);
      if (face && face.timestamp_seconds !== null) {
        videoRef.current.currentTime = face.timestamp_seconds;
      }
    }
  }, [selectedFrame, analysisData]);

  if (!activeEvidence) {
    return (
      <div className="space-y-6 max-w-7xl mx-auto">
        <div className="pb-4 border-b border-borderLight">
          <h2 className="text-xl font-mono font-bold tracking-tight text-textMain">
            FORENSIC ANALYSIS WORKSPACE
          </h2>
          <p className="text-xs text-textMuted font-mono mt-1 uppercase tracking-wide">
            Inspect Media Streams, Preprocess Bounding Boxes, & Run Neural Analyses
          </p>
        </div>

        <div className="p-12 bg-panelBg border border-borderLight rounded flex flex-col items-center justify-center text-center space-y-4 min-h-[400px]">
          <FileQuestion className="h-12 w-12 text-textMuted opacity-60" />
          <div className="font-mono space-y-1">
            <h3 className="text-xs font-bold text-textMain uppercase tracking-wider">
              NO ACTIVE EVIDENCE SELECTED
            </h3>
            <p className="text-xs text-textMuted max-w-sm">
              Please upload a media file on the Evidence page or select one from the Active Case list.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const hasAnalysis = analysisData !== null;
  const isCompleted = hasAnalysis && analysisData.status === "COMPLETED";
  const isFailed = hasAnalysis && analysisData.status === "FAILED";
  const activeFaces = getActiveFaces();

  const framesWithFaces = analysisData && analysisData.faces 
    ? Array.from(new Set(analysisData.faces.map(f => f.frame_index))).sort((a,b) => a-b)
    : [];

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Title */}
      <div className="pb-4 border-b border-borderLight flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h2 className="text-xl font-mono font-bold tracking-tight text-textMain uppercase">
            Forensic Workspace: {activeEvidence.original_filename}
          </h2>
          <p className="text-xs text-textMuted font-mono mt-1 uppercase tracking-wide">
            Media Preprocessing, Frame Sampling, and Face Localization
          </p>
        </div>
        <div className="flex items-center space-x-3">
          {isCompleted && (
            <div className="flex items-center space-x-2">
              <input
                id="show-boxes"
                type="checkbox"
                checked={showFaceBoxes}
                onChange={(e) => setShowFaceBoxes(e.target.checked)}
                className="rounded bg-darkBg border-borderLight text-forensicCyan focus:ring-0 focus:ring-offset-0 h-4 w-4 cursor-pointer"
              />
              <label htmlFor="show-boxes" className="text-xs font-mono text-textMuted uppercase cursor-pointer select-none">
                SHOW FACE BOXES
              </label>
            </div>
          )}
        </div>
      </div>

      {/* Main Split Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* LEFT: Viewport Stream */}
        <div className="lg:col-span-7 xl:col-span-8 flex flex-col space-y-4">
          <div className="bg-panelHeader border border-borderLight px-4 py-2 flex items-center justify-between">
            <span className="text-[10px] font-mono text-textMuted uppercase tracking-wider font-semibold">
              {activeEvidence.media_category === "VIDEO" ? "Sampled Frame Viewport" : "Static Image Viewport"}
            </span>
            <span className="text-[9px] font-mono text-textMuted uppercase">
              {isCompleted ? "PREPROCESSED REPRESENTATION" : "UNPROCESSED EVIDENCE"}
            </span>
          </div>

          <div className="relative bg-darkBg border border-borderLight rounded flex flex-col items-center justify-center p-4 min-h-[400px] overflow-hidden">
            {activeEvidence.media_category === "IMAGE" ? (
              /* IMAGE PREVIEW WITH BOX OVERLAYS */
              <div className="relative max-w-full max-h-[500px]">
                <img
                  ref={imgRef}
                  src={apiService.getEvidencePreviewUrl(activeEvidence.id)}
                  alt="Original Evidence Preview"
                  onLoad={handleImageLoad}
                  className="max-w-full max-h-[500px] object-contain rounded"
                />
                
                {/* Absolute overlay boxes scaled responsive using percent coordinates */}
                {isCompleted && showFaceBoxes && naturalWidth > 0 && naturalHeight > 0 && (
                  <div className="absolute inset-0">
                    {activeFaces.map((face) => {
                      const left = (face.x / naturalWidth) * 100;
                      const top = (face.y / naturalHeight) * 100;
                      const width = (face.width / naturalWidth) * 100;
                      const height = (face.height / naturalHeight) * 100;

                      return (
                        <div
                          key={face.id}
                          className="absolute border border-forensicCyan bg-forensicCyan/10 shadow-[0_0_8px_rgba(0,240,255,0.4)] transition-all duration-150"
                          style={{
                            left: `${left}%`,
                            top: `${top}%`,
                            width: `${width}%`,
                            height: `${height}%`
                          }}
                        >
                          <span className="absolute -top-4.5 -left-[1px] bg-forensicCyan text-darkBg text-[8px] font-mono font-bold px-1 py-0.5 rounded-t">
                            FACE_{face.face_index}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            ) : (
              /* VIDEO PREVIEW WITH FRAME INDEX SEEKING */
              <div className="relative max-w-full w-[640px]">
                <video
                  ref={videoRef}
                  src={apiService.getEvidencePreviewUrl(activeEvidence.id)}
                  controls
                  className="w-full rounded border border-borderLight"
                />
                
                {/* Visual Label if selected timestamp seeking */}
                {selectedFrame !== null && (
                  <div className="absolute top-2 left-2 bg-darkBg/85 border border-borderActive px-2.5 py-1 rounded text-[9px] font-mono text-forensicCyan tracking-wide">
                    SAMPLED FRAME: #{selectedFrame} • t = {(selectedFrame / (analysisData?.video_metadata?.fps || 30)).toFixed(2)}s
                  </div>
                )}

                {/* Draw face boxes over video container if we know dimensions */}
                {isCompleted && showFaceBoxes && selectedFrame !== null && analysisData.video_metadata && (
                  <div className="absolute inset-0 pointer-events-none">
                    {activeFaces.map((face) => {
                      const left = (face.x / analysisData.video_metadata.width) * 100;
                      const top = (face.y / analysisData.video_metadata.height) * 100;
                      const width = (face.width / analysisData.video_metadata.width) * 100;
                      const height = (face.height / analysisData.video_metadata.height) * 100;

                      return (
                        <div
                          key={face.id}
                          className="absolute border border-forensicCyan bg-forensicCyan/10 shadow-[0_0_8px_rgba(0,240,255,0.4)] pointer-events-auto"
                          style={{
                            left: `${left}%`,
                            top: `${top}%`,
                            width: `${width}%`,
                            height: `${height}%`
                          }}
                        >
                          <span className="absolute -top-4.5 -left-[1px] bg-forensicCyan text-darkBg text-[8px] font-mono font-bold px-1 py-0.5 rounded-t">
                            F_{face.face_index}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* RIGHT: Analysis Control & Preprocessing Panel */}
        <div className="lg:col-span-5 xl:col-span-4 flex flex-col space-y-4">
          
          {/* PANEL 1: Preprocessing Controls */}
          <div className="flex flex-col space-y-3">
            <div className="bg-panelHeader border border-borderLight px-4 py-2 flex items-center">
              <span className="text-[10px] font-mono text-textMuted uppercase tracking-wider font-semibold">
                Media Preprocessing Controller
              </span>
            </div>

            <div className="bg-panelBg border border-borderLight rounded-lg p-5 space-y-6">
              
              {/* Status indicators */}
              <div className="space-y-4 font-mono text-xs">
                <div className="flex justify-between items-center py-1.5 border-b border-borderLight/40">
                  <span className="text-textMuted">SHA-256 INTEGRITY:</span>
                  <span className={`font-bold ${isCompleted ? "text-forensicGreen" : hasAnalysis && !analysisData.integrity_verified ? "text-forensicRed" : "text-textMuted"}`}>
                    {isCompleted ? "VERIFIED" : hasAnalysis && !analysisData.integrity_verified ? "FAILED" : "PENDING"}
                  </span>
                </div>

                <div className="flex justify-between items-center py-1.5 border-b border-borderLight/40">
                  <span className="text-textMuted">PREPROCESSING:</span>
                  <span className={`font-bold ${isCompleted ? "text-forensicGreen" : isFailed ? "text-forensicRed" : isProcessing ? "text-forensicCyan animate-pulse" : "text-textMuted"}`}>
                    {isCompleted ? "COMPLETED" : isFailed ? "FAILED" : isProcessing ? "PROCESSING..." : "UNINITIALIZED"}
                  </span>
                </div>

                <div className="flex justify-between items-center py-1.5 border-b border-borderLight/40">
                  <span className="text-textMuted">FACE DETECTOR:</span>
                  <span className="text-textMain font-bold">HAAR CASCADE</span>
                </div>

                {isCompleted && (
                  <div className="flex justify-between items-center py-1.5 border-b border-borderLight/40">
                    <span className="text-textMuted">TOTAL FACES FOUND:</span>
                    <span className="text-forensicCyan font-bold">{analysisData.faces_detected}</span>
                  </div>
                )}
              </div>

              {/* Video Metadata display panel */}
              {isCompleted && analysisData.video_metadata && (
                <div className="bg-darkBg border border-borderLight p-3.5 rounded space-y-2 font-mono text-[11px]">
                  <span className="text-[9px] text-textMuted uppercase tracking-wider block font-bold">
                    SAMPLED VIDEO METADATA
                  </span>
                  <div className="grid grid-cols-2 gap-2 text-textMuted">
                    <div>Resolution: <span className="text-textMain">{analysisData.video_metadata.width}x{analysisData.video_metadata.height}</span></div>
                    <div>FPS: <span className="text-textMain">{analysisData.video_metadata.fps}</span></div>
                    <div>Duration: <span className="text-textMain">{analysisData.video_metadata.duration_seconds}s</span></div>
                    <div>Samples: <span className="text-textMain">{analysisData.video_metadata.sampled_frames} frames</span></div>
                  </div>
                </div>
              )}

              {/* Preprocessing Actions */}
              <div className="space-y-3.5 pt-2">
                <button
                  onClick={handleStartPreprocessing}
                  disabled={isProcessing || isAnalyzingDeepfake}
                  className={`
                    w-full py-3 text-xs font-mono font-bold uppercase tracking-wider rounded transition-all cursor-pointer flex items-center justify-center space-x-2.5
                    ${isProcessing 
                      ? "bg-borderActive text-textMuted border border-borderLight cursor-not-allowed" 
                      : "bg-forensicCyan text-darkBg hover:bg-white"
                    }
                  `}
                >
                  {isProcessing ? (
                    <>
                      <Activity className="h-4.5 w-4.5 animate-spin" />
                      <span>RUNNING PIPELINE...</span>
                    </>
                  ) : (
                    <>
                      <Cpu className="h-4.5 w-4.5" />
                      <span>START PREPROCESSING</span>
                    </>
                  )}
                </button>

                {processingError && (
                  <div className="bg-darkBg border border-forensicRed/20 p-3.5 rounded flex items-start space-x-2.5">
                    <AlertTriangle className="h-4.5 w-4.5 text-forensicRed flex-shrink-0 mt-0.5" />
                    <span className="text-[11px] font-mono text-textMuted leading-relaxed">
                      {processingError}
                    </span>
                  </div>
                )}
              </div>

            </div>
          </div>

          {/* PANEL 2: Deepfake Detection Controller */}
          <div className="flex flex-col space-y-3">
            <div className="bg-panelHeader border border-borderLight px-4 py-2 flex items-center">
              <span className="text-[10px] font-mono text-textMuted uppercase tracking-wider font-semibold">
                DEEPFAKE DETECTION
              </span>
            </div>

            <div className="bg-panelBg border border-borderLight rounded-lg p-5 space-y-5">
              
              {/* Deepfake status metrics */}
              <div className="space-y-4 font-mono text-xs">
                <div className="flex justify-between items-center py-1.5 border-b border-borderLight/40">
                  <span className="text-textMuted">MODEL NAME:</span>
                  <span className="text-textMain font-bold">{deepfakeData?.model_name || "MesoInception-4"}</span>
                </div>

                <div className="flex justify-between items-center py-1.5 border-b border-borderLight/40">
                  <span className="text-textMuted">MODEL VERSION:</span>
                  <span className="text-textMain font-bold">{deepfakeData?.model_version || "NOT RUN"}</span>
                </div>

                <div className="flex justify-between items-center py-1.5 border-b border-borderLight/40">
                  <span className="text-textMuted">RESULT STATE:</span>
                  <span className="text-forensicAmber font-bold uppercase">
                    {deepfakeData ? deepfakeData.overall_label.replaceAll("_", " ") : "NOT RUN"}
                  </span>
                </div>

                {deepfakeData && (
                  <>
                    <div className="flex justify-between items-center py-1.5 border-b border-borderLight/40">
                      <span className="text-textMuted">RUN STATUS:</span>
                      <span className="text-textMain font-bold">{deepfakeData.status}</span>
                    </div>
                    {deepfakeData.error_message && (
                      <p role="alert" className="text-forensicAmber">{deepfakeData.error_message}</p>
                    )}
                    <div className="flex justify-between items-center py-1.5 border-b border-borderLight/40">
                      <span className="text-textMuted">OVERALL RESULT:</span>
                      <span className={`font-bold ${deepfakeData.overall_label === "MODEL_NOT_CONFIGURED" ? "text-forensicAmber" : deepfakeData.overall_label === "NO_FACE_DETECTED" ? "text-textMuted" : "text-forensicCyan"}`}>
                        {deepfakeData.overall_label === "MODEL_NOT_CONFIGURED" ? "MODEL NOT CONFIGURED" : deepfakeData.overall_label.replace("_", " ")}
                      </span>
                    </div>

                    <div className="flex justify-between items-center py-1.5 border-b border-borderLight/40">
                      <span className="text-textMuted">CONFIDENCE INDEX:</span>
                      <span className="text-textMain font-bold">
                        {deepfakeData.overall_confidence !== null ? `${(deepfakeData.overall_confidence * 100).toFixed(1)}%` : "—"}
                      </span>
                    </div>
                  </>
                )}
              </div>

              {/* Start deepfake analyzer button */}
              <div className="space-y-3 pt-1">
                <button
                  onClick={handleStartDeepfakeAnalysis}
                  disabled={!isCompleted || isProcessing || isAnalyzingDeepfake}
                  className={`
                    w-full py-3 text-xs font-mono font-bold uppercase tracking-wider rounded transition-all cursor-pointer flex items-center justify-center space-x-2.5
                    ${(!isCompleted || isProcessing || isAnalyzingDeepfake)
                      ? "bg-borderActive text-textMuted border border-borderLight cursor-not-allowed" 
                      : "bg-forensicCyan text-darkBg hover:bg-white"
                    }
                  `}
                >
                  {isAnalyzingDeepfake ? (
                    <>
                      <Activity className="h-4.5 w-4.5 animate-spin" />
                      <span>ANALYZING...</span>
                    </>
                  ) : (
                    <>
                      <BrainCircuit className="h-4.5 w-4.5" />
                      <span>START DEEPFAKE ANALYSIS</span>
                    </>
                  )}
                </button>

                {!isCompleted && (
                  <span className="text-[9px] font-mono text-textMuted block text-center leading-relaxed">
                    Start preprocessing first to scan face regions before running neural check.
                  </span>
                )}

                {deepfakeError && (
                  <div className="bg-darkBg border border-forensicRed/20 p-3.5 rounded flex items-start space-x-2.5">
                    <AlertTriangle className="h-4.5 w-4.5 text-forensicRed flex-shrink-0 mt-0.5" />
                    <span className="text-[11px] font-mono text-textMuted leading-relaxed">
                      {deepfakeError}
                    </span>
                  </div>
                )}
              </div>

            </div>
          </div>

        </div>

      </div>

      {/* Frame Selection Menu for Videos */}
      {isCompleted && activeEvidence.media_category === "VIDEO" && framesWithFaces.length > 0 && (
        <div className="bg-panelBg border border-borderLight p-5 rounded-lg space-y-3">
          <span className="text-[10px] font-mono text-textMuted uppercase tracking-wider font-bold block">
            Select Sampled Frame containing Faces
          </span>
          <div className="flex flex-wrap gap-2">
            {framesWithFaces.map((frameIdx) => {
              const facesOnFrame = analysisData.faces.filter(f => f.frame_index === frameIdx).length;
              const isSelected = selectedFrame === frameIdx;
              return (
                <button
                  key={frameIdx}
                  onClick={() => setSelectedFrame(frameIdx)}
                  className={`
                    px-3 py-1.5 rounded text-xs font-mono border transition-all cursor-pointer
                    ${isSelected 
                      ? "bg-darkBg border-forensicCyan text-forensicCyan" 
                      : "bg-darkBg/60 border-borderLight text-textMuted hover:text-textMain hover:border-borderActive"
                    }
                  `}
                >
                  Frame #{frameIdx} ({facesOnFrame} face{facesOnFrame > 1 ? "s" : ""})
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* DETECTED FACE CROPS & INDIVIDUAL PREDICTIONS GRID */}
      {isCompleted && (
        <div className="bg-panelBg border border-borderLight rounded-lg p-6 space-y-4">
          <h3 className="text-xs font-mono font-bold text-textMain tracking-widest uppercase">
            DETECTED FACE CROPS & NEURAL PREDICTIONS
          </h3>

          {gradCamError && (
            <div className="p-3 bg-darkBg border border-forensicAmber/40 text-forensicAmber text-[10px] font-mono rounded">
              {gradCamError}
            </div>
          )}

          {analysisData.faces_detected > 0 ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
              {analysisData.faces.map((face) => {
                const prediction = deepfakeData && deepfakeData.predictions
                  ? deepfakeData.predictions.find(p => p.face_detection_id === face.id)
                  : null;

                return (
                  <div 
                    key={face.id}
                    className="bg-darkBg border border-borderLight rounded overflow-hidden p-2.5 flex flex-col space-y-2.5 hover:border-borderActive transition-colors"
                  >
                    <div className="h-28 bg-panelBg border border-borderLight/60 rounded flex items-center justify-center overflow-hidden">
                      <img
                        src={apiService.getFaceCropUrl(face.crop_path)}
                        alt={`Face ${face.face_index}`}
                        className="h-full w-full object-cover"
                      />
                    </div>

                    <div className="font-mono text-[9px] space-y-1">
                      <div className="text-textMain font-bold">Face #{face.face_index}</div>
                      {face.frame_index !== null && (
                        <div className="text-textMuted">Frame: #{face.frame_index}</div>
                      )}
                      
                      <div className="pt-1.5 border-t border-borderLight/30 text-textMuted space-y-1">
                        <div>Label: <span className={`font-semibold ${prediction ? "text-forensicAmber" : "text-textMuted"}`}>
                          {prediction ? (prediction.label === "MODEL_NOT_CONFIGURED" ? "UNCONFIGURED" : prediction.label) : "—"}
                        </span></div>
                        <div>Confidence: <span className="text-textMain font-bold">
                          {prediction && prediction.confidence !== null ? `${(prediction.confidence * 100).toFixed(1)}%` : "—"}
                        </span></div>
                        {prediction && (
                          <div className="text-[8px] text-textMuted">Status: {prediction.model_status.replace("_", " ")}</div>
                        )}

                        {/* Grad-CAM Explainability Overlay or Button */}
                        <div className="pt-2 border-t border-borderLight/30 flex flex-col space-y-1">
                          {gradCamMap[face.id] ? (
                            <div className="space-y-1">
                              <span className="text-[8px] text-forensicCyan font-bold uppercase block">Grad-CAM Overlay:</span>
                              <img
                                src={apiService.getGradCamImageUrl(gradCamMap[face.id])}
                                alt={`Grad-CAM ${face.id}`}
                                className="w-full h-20 object-cover rounded border border-forensicCyan/40"
                              />
                            </div>
                          ) : (
                            <button
                              type="button"
                              onClick={() => handleGenerateGradCam(face.id)}
                              disabled={isGeneratingGradCam === face.id}
                              className="w-full px-1.5 py-1 bg-darkBg border border-borderLight hover:border-forensicCyan/50 hover:text-forensicCyan text-[8px] font-mono uppercase tracking-wider rounded text-center transition-colors cursor-pointer disabled:opacity-40"
                            >
                              {isGeneratingGradCam === face.id ? "Generating..." : "Grad-CAM Heatmap"}
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="p-8 border border-dashed border-borderLight rounded text-center text-xs font-mono text-textMuted uppercase tracking-wide">
              NO FACES DETECTED
            </div>
          )}
        </div>
      )}

      {/* PHASE 5: FORENSIC SIGNAL ANALYSIS (ELA & FFT) */}
      <div className="bg-panelBg border border-borderLight rounded-lg overflow-hidden">
        {/* Panel Header & Navigation Tabs */}
        <div className="bg-panelHeader border-b border-borderLight px-6 py-1.5 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
          <span className="text-[10px] font-mono text-textMuted uppercase tracking-wider font-bold">
            Forensic Signal Analysis
          </span>
          <div className="flex space-x-2">
            <button
              onClick={() => setActiveSignalTab("ela")}
              className={`
                px-3 py-1.5 text-xs font-mono border transition-all cursor-pointer flex items-center space-x-1.5
                ${activeSignalTab === "ela" 
                  ? "bg-darkBg border-forensicCyan text-forensicCyan" 
                  : "bg-darkBg/30 border-transparent text-textMuted hover:text-textMain"
                }
              `}
            >
              <ImageIcon className="h-3.5 w-3.5" />
              <span>ERROR LEVEL (ELA)</span>
            </button>
            <button
              onClick={() => setActiveSignalTab("fft")}
              className={`
                px-3 py-1.5 text-xs font-mono border transition-all cursor-pointer flex items-center space-x-1.5
                ${activeSignalTab === "fft" 
                  ? "bg-darkBg border-forensicCyan text-forensicCyan" 
                  : "bg-darkBg/30 border-transparent text-textMuted hover:text-textMain"
                }
              `}
            >
              <Binary className="h-3.5 w-3.5" />
              <span>FFT COEFFS (2D FFT)</span>
            </button>
          </div>
        </div>

        {/* Tab Contents */}
        <div className="p-6">
          {activeEvidence.media_category !== "IMAGE" ? (
            /* Video Restriction Notice */
            <div className="p-8 text-center text-xs font-mono text-textMuted uppercase tracking-wide">
              Forensic signal features (ELA / FFT) are restricted to Image files.
            </div>
          ) : activeSignalTab === "ela" ? (
            /* --- ELA TAB --- */
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* ELA Left: Visualization viewport */}
                <div className="lg:col-span-7 xl:col-span-8 bg-darkBg border border-borderLight rounded flex flex-col items-center justify-center p-4 min-h-[300px]">
                  {elaData && elaData.status === "COMPLETED" && elaData.visualization_path ? (
                    <div className="relative max-w-full">
                      <img
                        src={apiService.getElaImageUrl(elaData.visualization_path)}
                        alt="Error Level Analysis Visualization"
                        className="max-w-full max-h-[450px] object-contain rounded"
                      />
                    </div>
                  ) : elaData && elaData.status === "PROCESSING" ? (
                    <div className="flex flex-col items-center space-y-3 font-mono text-xs text-forensicCyan">
                      <Activity className="h-8 w-8 animate-spin" />
                      <span className="animate-pulse">RECOMPRESSING MATRIX (Q=95)...</span>
                    </div>
                  ) : (
                    <div className="font-mono text-xs text-textMuted text-center py-10 uppercase tracking-wide">
                      Error Level Analysis Not Initiated
                    </div>
                  )}
                </div>

                {/* ELA Right: Details & Trigger controls */}
                <div className="lg:col-span-5 xl:col-span-4 bg-darkBg/40 border border-borderLight rounded p-5 space-y-5">
                  <span className="text-[10px] font-mono text-forensicCyan uppercase tracking-widest font-bold block">
                    JPEG Recompression Metrics
                  </span>

                  <div className="space-y-3 font-mono text-[11px] text-textMuted">
                    <div className="flex justify-between py-1 border-b border-borderLight/30">
                      <span>Status:</span>
                      <span className={`font-bold ${elaData && elaData.status === "COMPLETED" ? "text-forensicGreen" : elaData && elaData.status === "FAILED" ? "text-forensicRed" : "text-textMuted"}`}>
                        {elaData ? elaData.status : "UNINITIALIZED"}
                      </span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-borderLight/30">
                      <span>Config Quality:</span>
                      <span className="text-textMain font-bold">95 / 100</span>
                    </div>
                    {elaData && elaData.status === "COMPLETED" && (
                      <>
                        <div className="flex justify-between py-1 border-b border-borderLight/30">
                          <span>Mean Error:</span>
                          <span className="text-textMain font-bold">{elaData.mean_error}</span>
                        </div>
                        <div className="flex justify-between py-1 border-b border-borderLight/30">
                          <span>Max Error:</span>
                          <span className="text-textMain font-bold">{elaData.max_error}</span>
                        </div>
                        <div className="flex justify-between py-1 border-b border-borderLight/30">
                          <span>Error Variance:</span>
                          <span className="text-textMain font-bold">{elaData.error_variance}</span>
                        </div>
                        <div className="flex justify-between py-1 border-b border-borderLight/30">
                          <span>Completed At:</span>
                          <span className="text-textMain text-[10px]">
                            {new Date(elaData.completed_at).toLocaleTimeString()}
                          </span>
                        </div>
                      </>
                    )}
                  </div>

                  <div className="space-y-3 pt-2">
                    <button
                      onClick={handleStartElaAnalysis}
                      disabled={isAnalyzingEla}
                      className={`
                        w-full py-2.5 text-xs font-mono font-bold uppercase tracking-wider rounded transition-all cursor-pointer flex items-center justify-center space-x-2
                        ${isAnalyzingEla 
                          ? "bg-borderActive text-textMuted border border-borderLight cursor-not-allowed" 
                          : "bg-forensicCyan text-darkBg hover:bg-white"
                        }
                      `}
                    >
                      {isAnalyzingEla ? (
                        <>
                          <Activity className="h-4 w-4 animate-spin" />
                          <span>ANALYZING ELA...</span>
                        </>
                      ) : (
                        <>
                          <Cpu className="h-4 w-4" />
                          <span>RUN ERROR LEVEL ANALYSIS</span>
                        </>
                      )}
                    </button>

                    {elaError && (
                      <div className="bg-darkBg border border-forensicRed/20 p-3 rounded flex items-start space-x-2">
                        <AlertTriangle className="h-4 w-4 text-forensicRed flex-shrink-0 mt-0.5" />
                        <span className="text-[10px] font-mono text-textMuted leading-relaxed">
                          {elaError}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Forensic Disclaimer */}
              <div className="bg-darkBg/60 border border-borderLight p-4 rounded flex items-start space-x-3 text-[10px] font-mono text-textMuted leading-relaxed">
                <ShieldCheck className="h-5 w-5 text-forensicCyan flex-shrink-0" />
                <div>
                  <span className="text-textMain font-bold block mb-1">ELA INTERPRETATION SIGNAL GUIDE</span>
                  Error Level Analysis (ELA) resaves an image at a known quality level (95%) and compares it to the original. Gradients of error levels represent local compression coefficients. Higher variance indicates different compression stages or potential modifications. This is a statistical signal and does not constitute binary proof of image manipulation.
                </div>
              </div>
            </div>
          ) : (
            /* --- FFT TAB --- */
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* FFT Left: Spectrum viewport */}
                <div className="lg:col-span-7 xl:col-span-8 bg-darkBg border border-borderLight rounded flex flex-col items-center justify-center p-4 min-h-[300px]">
                  {fftData && fftData.status === "COMPLETED" && fftData.visualization_path ? (
                    <div className="relative max-w-full">
                      <img
                        src={apiService.getFftImageUrl(fftData.visualization_path)}
                        alt="2D FFT Magnitude Spectrum Visualization"
                        className="max-w-full max-h-[450px] object-contain rounded"
                      />
                    </div>
                  ) : fftData && fftData.status === "PROCESSING" ? (
                    <div className="flex flex-col items-center space-y-3 font-mono text-xs text-forensicCyan">
                      <Activity className="h-8 w-8 animate-spin" />
                      <span className="animate-pulse">COMPUTING 2D FOURIER TRANSFORM MATRIX...</span>
                    </div>
                  ) : (
                    <div className="font-mono text-xs text-textMuted text-center py-10 uppercase tracking-wide">
                      Fast Fourier Transform Not Initiated
                    </div>
                  )}
                </div>

                {/* FFT Right: Details & Trigger controls */}
                <div className="lg:col-span-5 xl:col-span-4 bg-darkBg/40 border border-borderLight rounded p-5 space-y-5">
                  <span className="text-[10px] font-mono text-forensicCyan uppercase tracking-widest font-bold block">
                    Frequency Domain Metrics
                  </span>

                  <div className="space-y-3 font-mono text-[11px] text-textMuted">
                    <div className="flex justify-between py-1 border-b border-borderLight/30">
                      <span>Status:</span>
                      <span className={`font-bold ${fftData && fftData.status === "COMPLETED" ? "text-forensicGreen" : fftData && fftData.status === "FAILED" ? "text-forensicRed" : "text-textMuted"}`}>
                        {fftData ? fftData.status : "UNINITIALIZED"}
                      </span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-borderLight/30">
                      <span>High-Freq Radius Threshold:</span>
                      <span className="text-textMain font-bold">15.0% of Diagonal</span>
                    </div>
                    {fftData && fftData.status === "COMPLETED" && (
                      <>
                        <div className="flex justify-between py-1 border-b border-borderLight/30">
                          <span>Average Magnitude:</span>
                          <span className="text-textMain font-bold">{fftData.average_magnitude}</span>
                        </div>
                        <div className="flex justify-between py-1 border-b border-borderLight/30">
                          <span>High-Freq Ratio:</span>
                          <span className="text-textMain font-bold">{(fftData.high_frequency_ratio * 100).toFixed(2)}%</span>
                        </div>
                        <div className="flex justify-between py-1 border-b border-borderLight/30">
                          <span>Spectral Entropy:</span>
                          <span className="text-textMain font-bold">{fftData.spectral_entropy}</span>
                        </div>
                        <div className="flex justify-between py-1 border-b border-borderLight/30">
                          <span>Completed At:</span>
                          <span className="text-textMain text-[10px]">
                            {new Date(fftData.completed_at).toLocaleTimeString()}
                          </span>
                        </div>
                      </>
                    )}
                  </div>

                  <div className="space-y-3 pt-2">
                    <button
                      onClick={handleStartFftAnalysis}
                      disabled={isAnalyzingFft}
                      className={`
                        w-full py-2.5 text-xs font-mono font-bold uppercase tracking-wider rounded transition-all cursor-pointer flex items-center justify-center space-x-2
                        ${isAnalyzingFft 
                          ? "bg-borderActive text-textMuted border border-borderLight cursor-not-allowed" 
                          : "bg-forensicCyan text-darkBg hover:bg-white"
                        }
                      `}
                    >
                      {isAnalyzingFft ? (
                        <>
                          <Activity className="h-4 w-4 animate-spin" />
                          <span>ANALYZING FFT...</span>
                        </>
                      ) : (
                        <>
                          <Cpu className="h-4 w-4" />
                          <span>RUN FAST FOURIER ANALYSIS</span>
                        </>
                      )}
                    </button>

                    {fftError && (
                      <div className="bg-darkBg border border-forensicRed/20 p-3 rounded flex items-start space-x-2">
                        <AlertTriangle className="h-4 w-4 text-forensicRed flex-shrink-0 mt-0.5" />
                        <span className="text-[10px] font-mono text-textMuted leading-relaxed">
                          {fftError}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Forensic Disclaimer */}
              <div className="bg-darkBg/60 border border-borderLight p-4 rounded flex items-start space-x-3 text-[10px] font-mono text-textMuted leading-relaxed">
                <ShieldCheck className="h-5 w-5 text-forensicCyan flex-shrink-0" />
                <div>
                  <span className="text-textMain font-bold block mb-1">FFT FREQUENCY DOMAIN SIGNAL GUIDE</span>
                  The Fast Fourier Transform (FFT) converts an image from the spatial domain to the frequency domain. The resulting magnitude spectrum shows zero-frequency components at the center, radiating outward to high-frequency components. Anomalies, sharp spikes, or concentric energy circles represent pattern regularity anomalies. High-frequency ratios measure texture density. This represents a frequency-domain feature and does not constitute binary proof of image manipulation.
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Forensic Honesty footer warning */}
      <div className="pt-2 border-t border-borderLight/40 text-[10px] font-mono text-textMuted leading-relaxed flex items-center space-x-2">
        <ShieldCheck className="h-4.5 w-4.5 text-forensicCyan" />
        <span>
          Veridoc Forensics reports objective statistical signals. Authenticity ratios, classification scores, and deepfake verification require an active neural pipeline configuration.
        </span>
      </div>

    </div>
  );
}
