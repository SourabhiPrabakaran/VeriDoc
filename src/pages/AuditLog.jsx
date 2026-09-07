import React, { useState, useEffect } from "react";
import { ClipboardList, Clock, Shield } from "lucide-react";
import { apiService } from "../services/api";

export default function AuditLog({ activeEvidence }) {
  const [logs, setLogs] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const fetchLogs = async () => {
      if (!activeEvidence) {
        setLogs([]);
        return;
      }

      setIsLoading(true);
      try {
        const auditData = await apiService.getAuditLog(activeEvidence.id);
        setLogs(auditData);
      } catch (err) {
        console.error("Failed to load audit logs", err);
        setLogs([]);
      } finally {
        setIsLoading(false);
      }
    };

    fetchLogs();
  }, [activeEvidence]);

  const activeFile = activeEvidence;

  return (
    <div className="space-y-6 max-w-3xl mx-auto">
      {/* Title */}
      <div className="pb-4 border-b border-borderLight">
        <h2 className="text-xl font-mono font-bold tracking-tight text-textMain">
          EVIDENCE AUDIT TRAIL
        </h2>
        <p className="text-xs text-textMuted font-mono mt-1 uppercase tracking-wide">
          Immutable Cryptographic Chain of Custody & Module Invocation Log
        </p>
      </div>

      {/* Main Container */}
      <div className="bg-panelBg border border-borderLight rounded-lg p-6 space-y-6">
        
        {/* Workspace Context Info */}
        {activeFile ? (
          <div className="bg-darkBg border border-borderActive rounded p-4 font-mono text-xs space-y-1">
            <span className="text-[10px] text-textMuted uppercase tracking-wider block">
              Active File Session:
            </span>
            <span className="text-textMain font-bold block truncate">
              {activeFile.original_filename}
            </span>
            <span className="text-textMuted text-[10px] block uppercase">
              Database ID: [{activeFile.id}] • SHA-256: {activeFile.sha256_hash.substring(0, 10)}...
            </span>
          </div>
        ) : (
          <div className="bg-darkBg border border-borderLight rounded p-4 flex items-center justify-between">
            <span className="text-xs font-mono text-textMuted uppercase">
              Workspace Context: Unregistered
            </span>
            <span className="h-2 w-2 rounded-full bg-textMuted"></span>
          </div>
        )}

        {/* Audit Timeline Shell */}
        <div className="border border-borderLight rounded-lg overflow-hidden">
          <div className="bg-panelHeader border-b border-borderLight px-4 py-2 flex items-center justify-between">
            <span className="text-[10px] font-mono text-textMuted uppercase tracking-wider font-semibold">
              Chain of Custody Timeline
            </span>
            <span className="text-[9px] font-mono text-textMuted uppercase">
              Scope: Database
            </span>
          </div>

          {isLoading ? (
            <div className="p-12 flex flex-col items-center justify-center space-y-2 text-textMuted font-mono text-xs">
              <span className="animate-spin h-5 w-5 border-2 border-forensicCyan border-t-transparent rounded-full"></span>
              <span className="animate-pulse text-[10px]">Loading Audit Log...</span>
            </div>
          ) : logs.length > 0 ? (
            /* Render Timeline Logs */
            <div className="p-6 bg-darkBg/20 space-y-6">
              {logs.map((log, index) => (
                <div key={log.id} className="relative flex items-start space-x-4 font-mono text-xs">
                  {/* Vertical connector line */}
                  {index < logs.length - 1 && (
                    <span 
                      className="absolute left-4.5 top-8 bottom-0 w-[1px] bg-borderLight" 
                      style={{ transform: "translateX(-50%)" }}
                    ></span>
                  )}
                  
                  {/* Event Icon Node */}
                  <div className="z-10 p-2 bg-panelBg border border-borderActive text-forensicCyan rounded">
                    <Shield className="h-4 w-4" />
                  </div>
                  
                  {/* Event Content */}
                  <div className="flex-1 space-y-1 bg-panelBg border border-borderLight rounded p-3">
                    <div className="flex justify-between items-center flex-wrap gap-2">
                      <span className="text-textMain font-bold text-[11px] uppercase tracking-wide">
                        {log.action}
                      </span>
                      <div className="flex items-center space-x-1.5 text-textMuted text-[10px]">
                        <Clock className="h-3 w-3" />
                        <span>{new Date(log.timestamp).toLocaleString()}</span>
                      </div>
                    </div>
                    
                    <div className="text-[11px] text-textMuted leading-relaxed pt-1 select-text">
                      {log.details}
                    </div>

                    <div className="text-[9px] text-textMuted pt-1.5 border-t border-borderLight/30 flex justify-between">
                      <span>Operator: <span className="text-textMain">ANALYST_0820</span></span>
                      <span>Log ID: #{log.id}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            /* Empty Audit Timeline State */
            <div className="p-12 flex flex-col items-center justify-center text-center space-y-4 min-h-[250px] bg-darkBg/20">
              <div className="p-3 bg-panelBg border border-borderLight rounded text-textMuted">
                <ClipboardList className="h-8 w-8" />
              </div>
              
              <div className="font-mono space-y-1">
                <h3 className="text-xs font-bold text-textMain uppercase tracking-wider">
                  NO AUDIT EVENTS
                </h3>
                <p className="text-xs text-textMuted max-w-sm">
                  Evidence activity will appear here after evidence registration.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Informative Note */}
        <div className="text-[10px] font-mono text-textMuted leading-relaxed bg-panelHeader border border-borderLight rounded p-3">
          <span className="text-textMain font-bold block mb-1 uppercase tracking-wider">
            Cryptographic Custody Note:
          </span>
          Every ingestion event, signature verification, and database transaction generates an immutable, timestamped record in the SQLite database to secure the forensic chain of custody.
        </div>

      </div>
    </div>
  );
}
