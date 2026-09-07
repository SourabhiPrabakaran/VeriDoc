import React, { useState, useEffect } from "react";
import { 
  FileBarChart2, 
  FileText, 
  Download, 
  AlertTriangle, 
  RefreshCw 
} from "lucide-react";
import { apiService } from "../services/api";

export default function Reports({ activeEvidence }) {
  const [reports, setReports] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [error, setError] = useState("");

  const activeFile = activeEvidence;

  const fetchReports = async () => {
    if (!activeEvidence) {
      setReports([]);
      return;
    }
    setIsLoading(true);
    setError("");
    try {
      const items = await apiService.getReports(activeEvidence.id);
      setReports(items);
    } catch (err) {
      console.error("Failed to load reports", err);
      setError("Failed to fetch generated reports list from database.");
      setReports([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchReports();
  }, [activeEvidence]);

  const handleGenerateReport = async () => {
    if (!activeEvidence) return;
    
    setIsGenerating(true);
    setStatusMessage("Compiling digital forensics telemetry...");
    setError("");
    
    try {
      await apiService.generateReport(activeEvidence.id);
      setStatusMessage("Report successfully registered in archive.");
      await fetchReports();
    } catch (err) {
      console.error("Failed to generate report", err);
      setError(err.message || "Failed to generate forensic report.");
    } finally {
      setIsGenerating(false);
      setStatusMessage("");
    }
  };

  const handleViewPdf = (reportUuid) => {
    const url = apiService.getReportDownloadUrl(reportUuid);
    window.open(url, "_blank", "noopener,noreferrer");
  };

  return (
    <div className="space-y-6 max-w-3xl mx-auto">
      {/* Title */}
      <div className="pb-4 border-b border-borderLight">
        <h2 className="text-xl font-mono font-bold tracking-tight text-textMain">
          FORENSIC REPORTS
        </h2>
        <p className="text-xs text-textMuted font-mono mt-1 uppercase tracking-wide">
          Export Documented Analysis Findings & Chain of Custody Certifications
        </p>
      </div>

      {/* Main Container */}
      <div className="bg-panelBg border border-borderLight rounded-lg p-6 space-y-6">
        
        {/* Workspace Ingest State */}
        {activeFile ? (
          <div className="bg-darkBg border border-borderActive rounded p-4 font-mono text-xs space-y-1">
            <span className="text-[10px] text-textMuted uppercase tracking-wider block">
              Active Assessment Context:
            </span>
            <span className="text-textMain font-bold block truncate">
              {activeFile.original_filename}
            </span>
            <span className="text-textMuted text-[10px] block uppercase">
              ID: [{activeFile.id}] • Status: {activeFile.status}
            </span>
          </div>
        ) : (
          <div className="bg-darkBg border border-borderLight rounded p-4 flex items-center justify-between">
            <span className="text-xs font-mono text-textMuted uppercase">
              Assessment Context: Waiting for evidence registration
            </span>
            <span className="h-2 w-2 rounded-full bg-textMuted"></span>
          </div>
        )}

        {/* Error Alert Box */}
        {error && (
          <div className="bg-darkBg border border-forensicRed/20 rounded p-4 flex items-start space-x-3.5 font-mono text-xs">
            <AlertTriangle className="h-5 w-5 text-forensicRed flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <h4 className="font-bold text-forensicRed uppercase tracking-wide">
                Execution Failure
              </h4>
              <p className="text-textMuted leading-relaxed select-text">{error}</p>
            </div>
          </div>
        )}

        {/* Loading Progress State */}
        {isGenerating && (
          <div className="bg-darkBg border border-borderActive rounded p-4 flex items-center space-x-3.5 font-mono text-xs text-textMuted animate-pulse">
            <RefreshCw className="h-4.5 w-4.5 text-forensicCyan animate-spin" />
            <span>{statusMessage}</span>
          </div>
        )}

        {/* Reports Panel Shell */}
        <div className="border border-borderLight rounded-lg overflow-hidden">
          <div className="bg-panelHeader border-b border-borderLight px-4 py-2 flex items-center justify-between">
            <span className="text-[10px] font-mono text-textMuted uppercase tracking-wider font-semibold">
              Available Case PDF Reports
            </span>
            <span className="text-[9px] font-mono text-textMuted uppercase">
              Engine: ReportLab
            </span>
          </div>

          {/* Conditional Rendering of Reports List */}
          {isLoading ? (
            <div className="p-12 flex flex-col items-center justify-center space-y-2 text-textMuted font-mono text-xs bg-darkBg/20">
              <RefreshCw className="h-5 w-5 text-forensicCyan animate-spin" />
              <span className="animate-pulse text-[10px]">Querying reports index...</span>
            </div>
          ) : !activeFile ? (
            <div className="p-12 flex flex-col items-center justify-center text-center space-y-4 min-h-[250px] bg-darkBg/20">
              <div className="p-3 bg-panelBg border border-borderLight rounded text-textMuted">
                <FileBarChart2 className="h-8 w-8" />
              </div>
              
              <div className="font-mono space-y-1">
                <h3 className="text-xs font-bold text-textMain uppercase tracking-wider">
                  NO ASSESSMENT CONTEXT
                </h3>
                <p className="text-xs text-textMuted max-w-sm">
                  Select or register an evidence file from the dashboard to review reports.
                </p>
              </div>
            </div>
          ) : reports.length > 0 ? (
            <div className="bg-darkBg/10 divide-y divide-borderLight">
              {reports.map((report) => (
                <div key={report.id} className="p-4 flex flex-col md:flex-row md:items-center md:justify-between gap-4 font-mono text-xs">
                  <div className="space-y-1.5 flex-1 min-w-0">
                    <div className="flex items-center space-x-2">
                      <span className="text-textMain font-bold uppercase tracking-wider text-[11px]">
                        Report #{report.id}
                      </span>
                      <span className={`px-2 py-0.5 rounded-[3px] text-[9px] font-bold tracking-wide uppercase ${
                        report.status === "COMPLETED" 
                          ? "bg-forensicGreen/10 border border-forensicGreen/20 text-forensicGreen"
                          : report.status === "FAILED"
                            ? "bg-forensicRed/10 border border-forensicRed/20 text-forensicRed"
                            : "bg-forensicCyan/10 border border-forensicCyan/20 text-forensicCyan animate-pulse"
                      }`}>
                        {report.status}
                      </span>
                    </div>
                    <div className="text-[10px] text-textMuted flex flex-col space-y-0.5">
                      <span className="truncate">UUID: <span className="text-textMain select-all">{report.report_uuid}</span></span>
                      {report.file_hash && (
                        <span className="truncate">SHA-256: <span className="text-forensicCyan select-all">{report.file_hash}</span></span>
                      )}
                      <span>Generated: <span className="text-textMain">{new Date(report.generated_at).toLocaleString()}</span></span>
                      {report.status === "FAILED" && report.error_message && (
                        <span className="text-forensicRed mt-1 block select-text">
                          Error: {report.error_message}
                        </span>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3 flex-shrink-0">
                    {report.status === "COMPLETED" && (
                      <button
                        onClick={() => handleViewPdf(report.report_uuid)}
                        className="px-3.5 py-2 bg-panelBg border border-borderLight hover:border-borderActive text-forensicCyan font-bold uppercase tracking-wider text-[10px] rounded transition-colors flex items-center space-x-1.5 cursor-pointer"
                      >
                        <Download className="h-3.5 w-3.5" />
                        <span>Download PDF</span>
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            /* Empty Reports State */
            <div className="p-12 flex flex-col items-center justify-center text-center space-y-4 min-h-[250px] bg-darkBg/20">
              <div className="p-3 bg-panelBg border border-borderLight rounded text-textMuted">
                <FileBarChart2 className="h-8 w-8" />
              </div>
              
              <div className="font-mono space-y-1">
                <h3 className="text-xs font-bold text-textMain uppercase tracking-wider">
                  NO REPORTS GENERATED
                </h3>
                <p className="text-xs text-textMuted max-w-sm">
                  Forensic reports can be generated only after evidence analysis has completed.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Informative Note */}
        <div className="text-[10px] font-mono text-textMuted leading-relaxed bg-panelHeader border border-borderLight rounded p-3">
          <span className="text-textMain font-bold block mb-1 uppercase tracking-wider">
            PDF Compilation Engine Note:
          </span>
          In Phase 3, the Generate Report utility compiles analysis telemetry, integrity hash verification logs, and frame annotations into an official judicial-grade PDF dossier using the python-ReportLab package.
        </div>

        {/* Action Panel */}
        <div className="flex justify-end pt-2">
          <button
            onClick={handleGenerateReport}
            disabled={!activeFile || isGenerating}
            className={`
              px-5 py-2.5 rounded font-mono text-xs font-bold uppercase tracking-wider flex items-center space-x-2 transition-colors
              ${!activeFile || isGenerating 
                ? "bg-borderLight text-textMuted border border-borderLight cursor-not-allowed" 
                : "bg-forensicCyan text-darkBg cursor-pointer hover:bg-white"
              }
            `}
          >
            {isGenerating ? (
              <>
                <RefreshCw className="h-4 w-4 animate-spin text-darkBg" />
                <span>GENERATING...</span>
              </>
            ) : (
              <>
                <FileText className="h-4 w-4" />
                <span>GENERATE REPORT</span>
              </>
            )}
          </button>
        </div>

      </div>
    </div>
  );
}
