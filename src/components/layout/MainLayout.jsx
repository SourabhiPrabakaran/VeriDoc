import React, { useState, useEffect } from "react";
import { 
  LayoutDashboard, 
  UploadCloud, 
  Cpu, 
  ShieldAlert, 
  FileSpreadsheet, 
  FileBarChart2, 
  Menu, 
  X, 
  User, 
  Activity,
  CheckCircle,
  AlertCircle,
  FileIcon,
  ChevronDown
} from "lucide-react";
import { apiService } from "../../services/api";

export default function MainLayout({ 
  activeView, 
  setActiveView, 
  activeEvidence, 
  setActiveEvidence, 
  children 
}) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isHealthCheckOpen, setIsHealthCheckOpen] = useState(false);
  
  // Real health check states
  const [systemState, setSystemState] = useState("PENDING");
  const [healthMessage, setHealthMessage] = useState("");
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);

  // Evidence list state for active case file selection
  const [evidenceItems, setEvidenceItems] = useState([]);

  const navigation = [
    { name: "Dashboard", id: "dashboard", icon: LayoutDashboard },
    { name: "Evidence", id: "evidence", icon: UploadCloud },
    { name: "Analysis", id: "analysis", icon: Cpu },
    { name: "Integrity", id: "integrity", icon: ShieldAlert },
    { name: "Audit Log", id: "audit", icon: FileSpreadsheet },
    { name: "Reports", id: "reports", icon: FileBarChart2 },
  ];

  // Run health check on mount and periodically
  const runHealthProbe = async () => {
    setIsCheckingHealth(true);
    const health = await apiService.healthCheck();
    setSystemState(health.status);
    setHealthMessage(health.message);
    setIsCheckingHealth(false);
  };

  // Fetch list of registered evidence to populate active selector
  const fetchEvidenceList = async () => {
    try {
      const items = await apiService.getEvidence();
      setEvidenceItems(items);
      
      // Auto-select first item if none is currently active and list has entries
      if (items.length > 0 && !activeEvidence) {
        setActiveEvidence(items[0]);
      }
    } catch (err) {
      console.error("Failed to retrieve evidence list", err);
    }
  };

  useEffect(() => {
    runHealthProbe();
    fetchEvidenceList();

    // Poll health status every 10 seconds to detect server state changes
    const healthInterval = setInterval(runHealthProbe, 10000);
    // Poll evidence list changes every 5 seconds to sync uploads
    const listInterval = setInterval(fetchEvidenceList, 5000);

    return () => {
      clearInterval(healthInterval);
      clearInterval(listInterval);
    };
  }, [activeEvidence]);

  const handleActiveFileChange = (e) => {
    const selectedId = parseInt(e.target.value, 10);
    const selected = evidenceItems.find(item => item.id === selectedId);
    if (selected) {
      setActiveEvidence(selected);
    }
  };

  const currentNav = navigation.find(n => n.id === activeView) || navigation[0];
  const isOnline = systemState === "ONLINE";

  return (
    <div className="min-h-screen bg-darkBg text-textMain flex flex-col md:flex-row font-sans">
      {/* MOBILE HEADER */}
      <div className="md:hidden flex items-center justify-between p-4 bg-panelBg border-b border-borderLight z-30">
        <div className="flex items-center space-x-2">
          <Activity className={`h-5 w-5 ${isOnline ? "text-forensicCyan animate-pulse" : "text-forensicRed"}`} />
          <span className="font-mono font-bold tracking-widest text-sm text-textMain">
            VERIDOC <span className="text-forensicCyan">FORENSICS</span>
          </span>
        </div>
        <button 
          onClick={() => setIsSidebarOpen(!isSidebarOpen)}
          className="p-1 text-textMuted hover:text-textMain hover:bg-borderLight rounded transition-colors"
        >
          {isSidebarOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </button>
      </div>

      {/* LEFT SIDEBAR */}
      <aside className={`
        fixed inset-y-0 left-0 transform md:relative md:translate-x-0 transition-transform duration-200 ease-in-out
        w-64 bg-panelBg border-r border-borderLight flex flex-col justify-between z-40
        ${isSidebarOpen ? "translate-x-0" : "-translate-x-full"}
      `}>
        <div>
          {/* Logo Brand */}
          <div className="p-6 border-b border-borderLight flex items-center space-x-3 hidden md:flex">
            <div className="p-2 bg-darkBg border border-borderLight rounded">
              <Activity className={`h-6 w-6 ${isOnline ? "text-forensicCyan animate-pulse" : "text-forensicRed"}`} />
            </div>
            <div className="flex flex-col">
              <span className="font-mono font-bold tracking-widest text-sm leading-none text-textMain">
                VERIDOC
              </span>
              <span className="font-mono text-xs font-semibold tracking-wider text-forensicCyan mt-1 leading-none">
                FORENSICS
              </span>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="p-4 space-y-1.5" aria-label="Main Navigation">
            {navigation.map((item) => {
              const Icon = item.icon;
              const isActive = activeView === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => {
                    setActiveView(item.id);
                    setIsSidebarOpen(false);
                  }}
                  className={`
                    w-full flex items-center space-x-3 px-4 py-3 rounded text-sm font-mono tracking-wide transition-all duration-150 group
                    ${isActive 
                      ? "bg-darkBg border border-borderActive text-forensicCyan shadow-sm" 
                      : "text-textMuted hover:text-textMain hover:bg-darkBg/60 border border-transparent"
                    }
                  `}
                >
                  <Icon className={`h-4.5 w-4.5 transition-colors ${isActive ? "text-forensicCyan" : "text-textMuted group-hover:text-textMain"}`} />
                  <span>{item.name}</span>
                  {isActive && <span className="ml-auto w-1 h-3 bg-forensicCyan rounded-full"></span>}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Footer Info */}
        <div className="p-4 border-t border-borderLight bg-darkBg/30">
          <div className="flex items-center space-x-3">
            <div className="h-8 w-8 rounded-full bg-borderActive flex items-center justify-center text-textMuted">
              <User className="h-4.5 w-4.5" />
            </div>
            <div className="flex flex-col min-w-0">
              <span className="text-xs font-mono text-textMain truncate">ANALYST_0820</span>
              <span className="text-[10px] font-mono text-textMuted uppercase tracking-wider">
                {isOnline ? "SERVER CONNECTED" : "OFFLINE WORKSPACE"}
              </span>
            </div>
          </div>
        </div>
      </aside>

      {/* OVERLAY FOR MOBILE SIDEBAR */}
      {isSidebarOpen && (
        <div 
          onClick={() => setIsSidebarOpen(false)}
          className="fixed inset-0 bg-black/60 z-30 md:hidden"
        ></div>
      )}

      {/* MAIN CONTAINER */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* TOP BAR */}
        <header className="h-16 bg-panelBg border-b border-borderLight flex items-center justify-between px-6 z-10">
          <div className="flex items-center space-x-4 min-w-0">
            <div className="flex items-center space-x-2">
              <span className="text-[10px] font-mono text-textMuted uppercase tracking-widest hidden sm:inline">
                SECTION_
              </span>
              <h1 className="text-sm font-mono font-bold uppercase tracking-wider text-textMain">
                {currentNav.name}
              </h1>
            </div>

            {/* Active Evidence Dropdown Selector */}
            {evidenceItems.length > 0 && (
              <div className="flex items-center space-x-2 border-l border-borderLight pl-4">
                <span className="text-[10px] font-mono text-textMuted uppercase tracking-wider hidden lg:inline">
                  Active Case:
                </span>
                <div className="relative">
                  <select
                    value={activeEvidence ? activeEvidence.id : ""}
                    onChange={handleActiveFileChange}
                    className="bg-darkBg border border-borderLight rounded px-3 py-1 text-xs font-mono text-textMain outline-none hover:border-borderActive transition-colors cursor-pointer appearance-none pr-8 select-none"
                  >
                    {evidenceItems.map((item) => (
                      <option key={item.id} value={item.id}>
                        [{item.id}] {item.original_filename}
                      </option>
                    ))}
                  </select>
                  <ChevronDown className="absolute right-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-textMuted pointer-events-none" />
                </div>
              </div>
            )}
          </div>

          <div className="flex items-center space-x-4">
            {/* System Status Component */}
            <button
              onClick={() => {
                setIsHealthCheckOpen(true);
                runHealthProbe();
              }}
              className="flex items-center space-x-2.5 px-3 py-1.5 bg-darkBg border border-borderLight rounded hover:border-borderActive transition-colors group cursor-pointer"
            >
              <span className="flex h-2 w-2 relative">
                {isOnline && (
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-forensicGreen opacity-75"></span>
                )}
                <span className={`relative inline-flex rounded-full h-2 w-2 ${isOnline ? "bg-forensicGreen" : "bg-forensicRed"}`}></span>
              </span>
              <span className="text-[10px] font-mono text-textMuted group-hover:text-textMain transition-colors">
                SYSTEM: <span className={`font-semibold uppercase ${isOnline ? "text-forensicGreen" : "text-forensicRed"}`}>
                  {isOnline ? "ONLINE" : "OFFLINE"}
                </span>
              </span>
            </button>
          </div>
        </header>

        {/* WORKSPACE CONTENT */}
        <main className="flex-1 overflow-y-auto p-6 md:p-8">
          {children}
        </main>
      </div>

      {/* HEALTH CHECK DIAGNOSTIC MODAL */}
      {isHealthCheckOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-sm">
          <div className="w-full max-w-lg bg-panelBg border border-borderActive rounded shadow-2xl overflow-hidden">
            {/* Modal Header */}
            <div className="p-4 bg-panelHeader border-b border-borderLight flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Activity className="h-4.5 w-4.5 text-forensicCyan" />
                <span className="font-mono text-xs font-bold tracking-wider text-textMain">
                  SYSTEM_DIAGNOSTIC_UTILITY
                </span>
              </div>
              <button 
                onClick={() => setIsHealthCheckOpen(false)}
                className="text-textMuted hover:text-textMain p-1 rounded hover:bg-darkBg transition-all cursor-pointer"
              >
                <X className="h-4.5 w-4.5" />
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-6 space-y-4">
              <div className="space-y-1">
                <h3 className="text-sm font-mono font-bold text-textMain">
                  Connection State & Verification
                </h3>
                <p className="text-xs text-textMuted font-sans">
                  Probes the live workspace backend to check endpoint accessibility.
                </p>
              </div>

              {/* Status Display Area */}
              <div className="bg-darkBg border border-borderLight rounded p-4 font-mono text-xs space-y-3">
                <div className="flex justify-between items-center pb-2 border-b border-borderLight/40">
                  <span className="text-textMuted">Probe Address:</span>
                  <span className="text-forensicCyan font-mono">GET http://127.0.0.1:8000/api/health</span>
                </div>

                {isCheckingHealth ? (
                  <div className="py-4 flex flex-col items-center justify-center space-y-2 text-textMuted">
                    <span className="animate-spin h-5 w-5 border-2 border-forensicCyan border-t-transparent rounded-full"></span>
                    <span className="text-[10px] tracking-wider uppercase animate-pulse">Running Probe...</span>
                  </div>
                ) : (
                  <div className="space-y-3.5">
                    <div className="flex items-center justify-between">
                      <span className="text-textMuted">Server Connection:</span>
                      <div className="flex items-center space-x-1.5">
                        {isOnline ? (
                          <>
                            <CheckCircle className="h-4 w-4 text-forensicGreen" />
                            <span className="text-forensicGreen font-bold uppercase">ONLINE</span>
                          </>
                        ) : (
                          <>
                            <AlertCircle className="h-4 w-4 text-forensicRed" />
                            <span className="text-forensicRed font-bold uppercase">OFFLINE</span>
                          </>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex justify-between">
                      <span className="text-textMuted">Active Service:</span>
                      <span className="text-textMain">{isOnline ? "veridoc-backend" : "unknown"}</span>
                    </div>

                    <div className="pt-2 border-t border-borderLight/40 text-[11px] text-textMuted leading-relaxed">
                      <span className="text-textMain font-semibold block mb-1">Diagnostic Report:</span>
                      {healthMessage}
                    </div>
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="flex justify-end space-x-3 pt-2">
                <button
                  onClick={runHealthProbe}
                  disabled={isCheckingHealth}
                  className="px-4 py-2 border border-borderActive rounded font-mono text-xs text-textMain hover:bg-darkBg transition-colors disabled:opacity-50 cursor-pointer"
                >
                  {isCheckingHealth ? "Running..." : "Re-Check"}
                </button>
                <button
                  onClick={() => setIsHealthCheckOpen(false)}
                  className="px-4 py-2 bg-borderActive rounded font-mono text-xs text-textMain hover:bg-borderActive/75 transition-colors cursor-pointer"
                >
                  Dismiss
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
