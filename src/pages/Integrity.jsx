import React from "react";
import { 
  ShieldCheck, 
  Binary, 
  AlertTriangle 
} from "lucide-react";

export default function Integrity({ activeEvidence }) {
  const activeFile = activeEvidence;

  return (
    <div className="space-y-6 max-w-3xl mx-auto">
      {/* Title */}
      <div className="pb-4 border-b border-borderLight">
        <h2 className="text-xl font-mono font-bold tracking-tight text-textMain">
          EVIDENCE INTEGRITY
        </h2>
        <p className="text-xs text-textMuted font-mono mt-1 uppercase tracking-wide">
          Cryptographic Hash Matching & Forensic Chain of Custody Verification
        </p>
      </div>

      {/* Main Hash Pane */}
      <div className="bg-panelBg border border-borderLight rounded-lg p-6 space-y-6">
        
        {/* Workspace Context Info */}
        {activeFile ? (
          <div className="bg-darkBg border border-borderActive rounded p-4 font-mono text-xs space-y-1">
            <span className="text-[10px] text-textMuted uppercase tracking-wider block">
              Active File in Register:
            </span>
            <span className="text-textMain font-bold block truncate">
              {activeFile.original_filename}
            </span>
            <span className="text-textMuted text-[10px] block uppercase">
              Format: {activeFile.mime_type} • Size: {activeFile.file_size} bytes
            </span>
          </div>
        ) : (
          <div className="bg-darkBg border border-borderLight rounded p-4 flex items-center justify-between">
            <span className="text-xs font-mono text-textMuted uppercase">
              Workspace Context: No active file selected
            </span>
            <span className="h-2 w-2 rounded-full bg-textMuted"></span>
          </div>
        )}

        {/* Cryptographic Hash Blocks */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Original SHA-256 Card */}
          <div className="bg-darkBg border border-borderLight p-4 rounded space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono text-textMuted uppercase tracking-wider font-semibold">
                Original SHA-256
              </span>
              <Binary className="h-4 w-4 text-textMuted" />
            </div>
            <div className="bg-panelBg border border-borderLight/60 rounded px-3 py-2.5 font-mono text-xs text-textMuted min-h-[38px] break-all select-none">
              Not available
            </div>
            <span className="text-[9px] font-mono text-textMuted uppercase block">
              Registered at initial ingest
            </span>
          </div>

          {/* Current SHA-256 Card */}
          <div className="bg-darkBg border border-borderLight p-4 rounded space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono text-textMuted uppercase tracking-wider font-semibold">
                Current SHA-256
              </span>
              <Binary className="h-4 w-4 text-textMuted" />
            </div>
            <div className="bg-panelBg border border-borderLight/60 rounded px-3 py-2.5 font-mono text-xs text-textMuted min-h-[38px] break-all select-none">
              {activeFile ? activeFile.sha256_hash : "Not available"}
            </div>
            <span className="text-[9px] font-mono text-textMuted uppercase block">
              Calculated on active buffer
            </span>
          </div>
        </div>

        {/* Integrity Status Indicator */}
        <div className="space-y-2">
          <span className="text-[10px] font-mono text-textMuted uppercase tracking-wider font-semibold block">
            Integrity Check Status
          </span>
          <div className="bg-panelHeader border border-borderLight p-4 rounded flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="h-3.5 w-3.5 rounded-full bg-borderActive flex items-center justify-center">
                <span className={`h-1.5 w-1.5 rounded-full ${activeFile ? "bg-forensicCyan" : "bg-textMuted"}`}></span>
              </div>
              <span className={`text-xs font-mono font-bold uppercase tracking-wide ${activeFile ? "text-forensicCyan" : "text-textMuted"}`}>
                {activeFile ? "HASH GENERATED" : "WAITING FOR EVIDENCE"}
              </span>
            </div>
            <span className="text-[9px] font-mono text-textMuted uppercase">
              {activeFile ? "HASH ARCHIVED" : "Verification Offline"}
            </span>
          </div>
        </div>

        {/* Informational Alerts */}
        <div className="bg-panelHeader border border-borderLight rounded p-4 flex items-start space-x-3">
          <AlertTriangle className="h-5 w-5 text-forensicAmber flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="text-xs font-mono font-bold text-forensicAmber uppercase tracking-wider">
              Verification Engine Boundaries (Phase 2)
            </h4>
            <p className="text-xs text-textMuted leading-relaxed">
              We have computed the secure SHA-256 hash of the evidence file. Comparing this hash against an external source database or running active tampering checks will be added in future phases.
            </p>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex justify-end pt-2">
          <button
            disabled
            className="px-5 py-2.5 bg-borderActive text-textMuted border border-borderLight rounded font-mono text-xs font-bold uppercase tracking-wider cursor-not-allowed flex items-center space-x-2"
          >
            <ShieldCheck className="h-4 w-4" />
            <span>VERIFY INTEGRITY</span>
          </button>
        </div>

      </div>
    </div>
  );
}
