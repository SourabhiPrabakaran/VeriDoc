import React, { useState, useEffect } from "react";
import { 
  Upload, 
  Activity, 
  Database,
  AlertTriangle,
  FolderOpen
} from "lucide-react";
import { apiService } from "../services/api";

export default function Dashboard({ setActiveView, setActiveEvidence }) {
  const [stats, setStats] = useState({
    total_evidence: 0,
    analyses_completed: 0,
    high_risk_findings: 0,
    integrity_warnings: 0
  });

  useEffect(() => {
    let isMounted = true;
    const fetchDashboardData = async () => {
      try {
        const [items, liveStats] = await Promise.all([
          apiService.getEvidence(),
          apiService.getDashboardStats()
        ]);
        if (isMounted) {
          setStats(liveStats);
          if (items.length > 0) {
            setActiveEvidence(prev => prev || items[0]);
          }
        }
      } catch (err) {
        console.error("Dashboard failed to fetch live statistics", err);
      }
    };
    
    fetchDashboardData();
    return () => { isMounted = false; };
  }, [setActiveEvidence]);

  const summaryCards = [
    { 
      title: "TOTAL EVIDENCE", 
      value: String(stats.total_evidence), 
      icon: FolderOpen, 
      desc: "Registered digital media in vault" 
    },
    { 
      title: "ANALYSES COMPLETED", 
      value: String(stats.analyses_completed), 
      icon: Activity, 
      desc: "Completed deepfake, ELA & FFT runs" 
    },
    { 
      title: "HIGH-RISK FINDINGS", 
      value: String(stats.high_risk_findings), 
      icon: AlertTriangle, 
      desc: "Detected deepfakes and forensic anomalies" 
    },
    { 
      title: "INTEGRITY WARNINGS", 
      value: String(stats.integrity_warnings), 
      icon: Database, 
      desc: "Cryptographic hash mismatch events" 
    },
  ];

  const forensicModules = [
    {
      title: "DEEPFAKE DETECTION",
      status: "OPERATIONAL",
      desc: "PyTorch MesoInception-4 deep learning facial manipulation analysis with Grad-CAM explainability.",
      badgeColor: "bg-forensicGreen/10 text-forensicGreen border border-forensicGreen/20"
    },
    {
      title: "ERROR LEVEL ANALYSIS (ELA)",
      status: "OPERATIONAL",
      desc: "JPEG multi-pass compression artifact evaluation for detecting recompression and splice boundaries.",
      badgeColor: "bg-forensicGreen/10 text-forensicGreen border border-forensicGreen/20"
    },
    {
      title: "FFT SPECTRAL ANALYSIS",
      status: "OPERATIONAL",
      desc: "2D Fast Fourier Transform high-frequency radial distribution analysis for GAN/synthesis artifacts.",
      badgeColor: "bg-forensicGreen/10 text-forensicGreen border border-forensicGreen/20"
    },
    {
      title: "FACIAL LOCALIZATION",
      status: "OPERATIONAL",
      desc: "Facial region localization, bounding-box clamping, and frame sampling for video evidence.",
      badgeColor: "bg-forensicGreen/10 text-forensicGreen border border-forensicGreen/20"
    },
    {
      title: "METADATA & EXIF FORENSICS",
      status: "FUTURE EXTENSION",
      desc: "Structural container and EXIF provenance extraction (planned for upcoming development cycle).",
      badgeColor: "bg-textMuted/10 text-textMuted border border-borderLight"
    },
    {
      title: "DUAL-STREAM MULTIMODAL FUSION",
      status: "RESEARCH EXTENSION",
      desc: "Joint spatial-frequency neural network combining RGB crops with 2D Fourier magnitude spectra.",
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
