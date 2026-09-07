import React, { useState, useEffect } from "react";
import { 
  FileText, 
  Search, 
  Upload, 
  Activity, 
  Database,
  AlertTriangle,
  FolderOpen
} from "lucide-react";
import { apiService } from "../services/api";

export default function Dashboard({ setActiveView, setActiveEvidence }) {
  const [totalEvidence, setTotalEvidence] = useState(0);

  useEffect(() => {
    const fetchCount = async () => {
      try {
        const items = await apiService.getEvidence();
        setTotalEvidence(items.length);
        
        // Auto-initialize active context if items exist
        if (items.length > 0) {
          // Keep active evidence synchronized
          setActiveEvidence(prev => prev || items[0]);
        }
      } catch (err) {
        console.error("Dashboard failed to fetch evidence count", err);
      }
    };
    
    fetchCount();
  }, [setActiveEvidence]);

  const summaryCards = [
    { title: "TOTAL EVIDENCE", value: String(totalEvidence), icon: FolderOpen, desc: "Total registered files in vault index" },
    { title: "ANALYSES COMPLETED", value: "0", icon: Activity, desc: "No completed module runs" },
    { title: "HIGH-RISK FINDINGS", value: "0", icon: AlertTriangle, desc: "No detection results" },
    { title: "INTEGRITY WARNINGS", value: "0", icon: Database, desc: "Integrity checks not run" },
  ];

  const forensicModules = [
    {
      title: "DEEPFAKE DETECTION",
      status: "READY FOR EVIDENCE",
      desc: "Deep learning-based facial manipulation analysis using the planned deepfake detection model interface.",
      badgeColor: "bg-forensicGreen/10 text-forensicGreen border border-forensicGreen/20"
    },
    {
      title: "ELA ANALYSIS",
      status: "WAITING FOR MEDIA",
      desc: "Error Level Analysis for detecting potential recompression and manipulation inconsistencies.",
      badgeColor: "bg-textMuted/10 text-textMuted border border-borderLight"
    },
    {
      title: "FFT ANALYSIS",
      status: "WAITING FOR MEDIA",
      desc: "Frequency-domain analysis for identifying potential manipulation-related patterns.",
      badgeColor: "bg-textMuted/10 text-textMuted border border-borderLight"
    },
    {
      title: "METADATA ANALYSIS",
      status: "WAITING FOR MEDIA",
      desc: "EXIF and structural metadata extraction for evidence context and provenance analysis.",
      badgeColor: "bg-textMuted/10 text-textMuted border border-borderLight"
    },
    {
      title: "FACE ANALYSIS",
      status: "WAITING FOR MEDIA",
      desc: "Face detection and facial-region analysis for downstream deepfake assessment.",
      badgeColor: "bg-textMuted/10 text-textMuted border border-borderLight"
    },
    {
      title: "EVIDENCE FUSION",
      status: "WAITING FOR ANALYSIS",
      desc: "Consolidated assessment combining available forensic signals.",
      badgeColor: "bg-textMuted/10 text-textMuted border border-borderLight"
    }
  ];

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Title & Description Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 pb-6 border-b border-borderLight">
        <div>
          <h2 className="text-xl md:text-2xl font-mono font-bold tracking-tight text-textMain">
            FORENSIC ANALYSIS CENTER
          </h2>
          <p className="text-xs text-textMuted font-mono mt-1 uppercase tracking-wide">
            Digital Evidence Authentication & Analysis WORKSTATION
          </p>
        </div>
        <div>
          <button
            onClick={() => setActiveView("evidence")}
            className="flex items-center space-x-2.5 px-4 py-2 bg-forensicCyan text-darkBg font-mono text-xs font-bold uppercase tracking-wider rounded hover:bg-white transition-all cursor-pointer shadow-lg hover:shadow-forensicCyan/15"
          >
            <Upload className="h-4 w-4" />
            <span>UPLOAD EVIDENCE</span>
          </button>
        </div>
      </div>

      {/* Summary Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {summaryCards.map((card, i) => {
          const Icon = card.icon;
          return (
            <div 
              key={i} 
              className="bg-panelBg border border-borderLight p-5 rounded flex flex-col justify-between hover:border-borderActive transition-colors"
            >
              <div className="flex justify-between items-start">
                <span className="text-[10px] font-mono text-textMuted uppercase tracking-wider font-semibold">
                  {card.title}
                </span>
                <div className="p-1.5 bg-darkBg border border-borderLight rounded text-textMuted">
                  <Icon className="h-4.5 w-4.5" />
                </div>
              </div>
              <div className="mt-4">
                <span className="text-3xl font-mono font-bold tracking-tight text-textMain">
                  {card.value}
                </span>
                <span className="block text-[10px] text-textMuted font-mono mt-2 uppercase tracking-wide">
                  {card.desc}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Forensic Module Status Grid */}
      <div className="space-y-4">
        <h3 className="text-xs font-mono font-bold text-textMain tracking-widest uppercase">
          ANALYSIS CAPABILITIES & MODULE STATES
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {forensicModules.map((mod, i) => (
            <div 
              key={i} 
              className="bg-panelBg border border-borderLight rounded flex flex-col justify-between overflow-hidden hover:border-borderActive transition-all duration-150"
            >
              {/* Module Header */}
              <div className="p-4 bg-panelHeader/45 border-b border-borderLight/60 flex items-center justify-between">
                <h4 className="text-[11px] font-mono font-bold tracking-wide text-textMain">
                  {mod.title}
                </h4>
                <span className={`px-2 py-0.5 rounded text-[9px] font-mono font-bold tracking-wider uppercase ${mod.badgeColor}`}>
                  {mod.status}
                </span>
              </div>
              {/* Module Body */}
              <div className="p-4 flex-1 flex flex-col justify-between">
                <p className="text-xs text-textMuted leading-relaxed animate-none">
                  {mod.desc}
                </p>
                
                {/* Visual Empty Metric Bar */}
                <div className="mt-5 space-y-1.5">
                  <div className="flex justify-between text-[9px] font-mono text-textMuted uppercase">
                    <span>Active Telemetry Signal</span>
                    <span>--</span>
                  </div>
                  <div className="h-1 bg-darkBg border border-borderLight rounded-full overflow-hidden">
                    <div className="h-full bg-borderActive w-0"></div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
