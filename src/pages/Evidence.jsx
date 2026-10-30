import React, { useState, useRef } from "react";
import { 
  UploadCloud, 
  Video, 
  CheckCircle2, 
  X, 
  FileIcon,
  RefreshCw,
  AlertTriangle
} from "lucide-react";
import { apiService } from "../services/api";

export default function Evidence({ activeEvidence, setActiveEvidence }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("idle"); // idle, uploading, success, error
  const [statusMessage, setStatusMessage] = useState("");
  const fileInputRef = useRef(null);

  const supportedFormats = [
    ".jpg", ".jpeg", ".png", ".webp", 
    ".mp4", ".mov", ".avi"
  ];

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const processFile = (file) => {
    if (!file) return;

    // Perform basic client-side extension check
    const filename = file.name.toLowerCase();
    const isSupported = supportedFormats.some(ext => filename.endsWith(ext));

    if (!isSupported) {
      setUploadStatus("error");
      setStatusMessage(`Unsupported format. Supported: ${supportedFormats.join(", ")}`);
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
    setUploadStatus("idle");
    setStatusMessage("");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const handleUploadSubmit = async () => {
    if (!selectedFile) return;

    setUploadStatus("uploading");
    setStatusMessage("Ingesting file and generating SHA-256 cryptographic hash...");

    try {
      const result = await apiService.uploadEvidence(selectedFile);
      setUploadStatus("success");
      setStatusMessage("Evidence registration complete. SHA-256 integrity hash archived.");
      setActiveEvidence(result);
    } catch (err) {
      setUploadStatus("error");
      setStatusMessage(err.message || "An error occurred during evidence registration.");
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setUploadStatus("idle");
    setStatusMessage("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const formatBytes = (bytes) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  return (
    <div className="space-y-6 max-w-3xl mx-auto">
      {/* Title */}
      <div className="pb-4 border-b border-borderLight">
        <h2 className="text-xl font-mono font-bold tracking-tight text-textMain">
          EVIDENCE REGISTRATION
        </h2>
        <p className="text-xs text-textMuted font-mono mt-1 uppercase tracking-wide">
          Register Digital Media in Local Database & Vault
        </p>
      </div>

      {/* Main Upload Box */}
      <div className="bg-panelBg border border-borderLight rounded-lg p-6 space-y-6">
        
        {uploadStatus === "idle" && !selectedFile && (
          <div
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current.click()}
            className={`
              border-2 border-dashed rounded-lg p-10 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-150 min-h-[280px]
              ${isDragActive 
                ? "border-forensicCyan bg-forensicCyan/5" 
                : "border-borderLight hover:border-borderActive hover:bg-darkBg/30"
              }
            `}
          >
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileChange}
              accept=".jpg,.jpeg,.png,.webp,.mp4,.mov,.avi"
              className="hidden"
            />
            <UploadCloud className={`h-12 w-12 mb-4 transition-colors ${isDragActive ? "text-forensicCyan" : "text-textMuted"}`} />
            <span className="text-sm font-mono text-textMain font-bold tracking-wide">
              DRAG & DROP EVIDENCE FILE
            </span>
            <span className="text-xs text-textMuted mt-1 block font-sans">
              or click to search local directory
            </span>

            <div className="mt-8 pt-6 border-t border-borderLight/60 w-full max-w-md">
              <span className="text-[10px] font-mono text-textMuted uppercase tracking-wider block mb-2">
                Supported Formats
              </span>
              <p className="text-[10px] text-textMuted font-mono uppercase">
                Images: JPG, JPEG, PNG, WebP • Videos: MP4, MOV, AVI
              </p>
            </div>
          </div>
        )}

        {/* Selected File Details */}
        {selectedFile && uploadStatus === "idle" && (
          <div className="space-y-5">
            <div className="bg-darkBg border border-borderActive rounded p-4 flex items-start justify-between">
              <div className="flex items-center space-x-3.5">
                <div className="p-3 bg-panelBg border border-borderLight rounded text-forensicCyan">
                  {selectedFile.type.startsWith("video/") ? <Video className="h-6 w-6" /> : <FileIcon className="h-6 w-6" />}
                </div>
                <div className="font-mono text-xs space-y-1">
                  <div className="text-textMain font-bold truncate max-w-md">{selectedFile.name}</div>
                  <div className="text-textMuted">Size: {formatBytes(selectedFile.size)}</div>
                  <div className="text-textMuted">Type: {selectedFile.type || "unknown"}</div>
                </div>
              </div>
              
              <button
                onClick={handleReset}
                className="p-1.5 text-textMuted hover:text-forensicRed hover:bg-borderLight rounded transition-colors cursor-pointer"
              >
                <X className="h-4.5 w-4.5" />
              </button>
            </div>

            {/* Action Buttons */}
            <div className="flex justify-end space-x-3 pt-2">
              <button
                onClick={handleReset}
                className="px-4 py-2 border border-borderLight hover:border-borderActive text-xs font-mono text-textMuted hover:text-textMain rounded transition-colors cursor-pointer"
              >
                Clear
              </button>
              
              <button
                onClick={handleUploadSubmit}
                className="px-4 py-2 bg-forensicCyan text-darkBg font-mono text-xs font-bold uppercase tracking-wider rounded hover:bg-white transition-colors cursor-pointer"
              >
                Register Evidence
              </button>
            </div>
          </div>
        )}

        {/* Loading/Uploading Spinner */}
        {uploadStatus === "uploading" && (
          <div className="py-12 flex flex-col items-center justify-center space-y-4">
            <RefreshCw className="h-8 w-8 text-forensicCyan animate-spin" />
            <div className="text-center font-mono space-y-1">
              <span className="text-xs text-textMain block uppercase tracking-wide">
                Ingestion In Progress
              </span>
              <span className="text-[10px] text-textMuted block animate-pulse">
                {statusMessage}
              </span>
            </div>
          </div>
        )}

        {/* Success Screen: Displays Real Metadata */}
        {uploadStatus === "success" && activeEvidence && (
          <div className="space-y-6">
            <div className="bg-darkBg border border-borderActive rounded p-5 flex items-center space-x-4">
              <div className="p-3 bg-forensicGreen/10 border border-forensicGreen/20 text-forensicGreen rounded-full">
                <CheckCircle2 className="h-6 w-6" />
              </div>
              <div className="font-mono text-xs space-y-0.5">
                <h4 className="font-bold text-textMain uppercase tracking-wide">
                  Evidence Registered
                </h4>
                <p className="text-textMuted">{statusMessage}</p>
              </div>
            </div>

            {/* Metadata Fields */}
            <div className="bg-darkBg border border-borderLight rounded divide-y divide-borderLight font-mono text-xs">
              <div className="p-3 flex justify-between">
                <span className="text-textMuted">Evidence ID:</span>
                <span className="text-textMain font-bold">[{activeEvidence.id}]</span>
              </div>
              <div className="p-3 flex justify-between">
                <span className="text-textMuted">Filename:</span>
                <span className="text-textMain truncate max-w-sm" title={activeEvidence.original_filename}>
                  {activeEvidence.original_filename}
                </span>
              </div>
              <div className="p-3 flex justify-between">
                <span className="text-textMuted">Media Category:</span>
                <span className="text-forensicCyan font-bold uppercase">{activeEvidence.media_category}</span>
              </div>
              <div className="p-3 flex justify-between">
                <span className="text-textMuted">MIME Type:</span>
                <span className="text-textMain">{activeEvidence.mime_type}</span>
              </div>
              <div className="p-3 flex justify-between">
                <span className="text-textMuted">Size:</span>
                <span className="text-textMain">{formatBytes(activeEvidence.file_size)}</span>
              </div>
              <div className="p-3 flex flex-col space-y-1">
                <span className="text-textMuted">SHA-256 Hash:</span>
                <span className="text-forensicCyan break-all bg-panelBg/70 p-2 rounded border border-borderLight select-all">
                  {activeEvidence.sha256_hash}
                </span>
              </div>
              <div className="p-3 flex justify-between">
                <span className="text-textMuted">Status:</span>
                <span className="text-forensicGreen font-bold uppercase">{activeEvidence.status}</span>
              </div>
              <div className="p-3 flex justify-between">
                <span className="text-textMuted">Timestamp:</span>
                <span className="text-textMain">{new Date(activeEvidence.created_at).toLocaleString()}</span>
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <button
                onClick={handleReset}
                className="px-4 py-2 bg-borderActive rounded font-mono text-xs text-textMain hover:bg-borderActive/70 transition-colors cursor-pointer"
              >
                Register Another File
              </button>
            </div>
          </div>
        )}

        {/* Error Screen */}
        {uploadStatus === "error" && (
          <div className="space-y-6">
            <div className="bg-darkBg border border-forensicRed/20 rounded p-5 flex items-start space-x-3.5">
              <AlertTriangle className="h-5 w-5 text-forensicRed flex-shrink-0 mt-0.5" />
              <div className="font-mono text-xs space-y-1.5">
                <h4 className="font-bold text-forensicRed uppercase tracking-wide">
                  Registration Failed
                </h4>
                <p className="text-textMuted leading-relaxed">{statusMessage}</p>
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <button
                onClick={handleReset}
                className="px-4 py-2 bg-borderActive rounded font-mono text-xs text-textMain hover:bg-borderActive/70 transition-colors cursor-pointer"
              >
                Try Again
              </button>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
