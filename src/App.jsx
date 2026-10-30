import React, { useState, useEffect } from "react";
import MainLayout from "./components/layout/MainLayout";
import Dashboard from "./pages/Dashboard";
import Evidence from "./pages/Evidence";
import Analysis from "./pages/Analysis";
import Integrity from "./pages/Integrity";
import AuditLog from "./pages/AuditLog";
import Reports from "./pages/Reports";

const VALID_VIEWS = ["dashboard", "evidence", "analysis", "integrity", "audit", "reports"];

function getInitialView() {
  if (typeof window === "undefined") return "dashboard";
  const hash = window.location.hash.replace("#", "").toLowerCase();
  if (VALID_VIEWS.includes(hash)) return hash;
  const params = new URLSearchParams(window.location.search);
  const viewParam = params.get("view")?.toLowerCase();
  if (VALID_VIEWS.includes(viewParam)) return viewParam;
  return "dashboard";
}

function App() {
  const [activeView, setActiveViewState] = useState(getInitialView);
  // Holds metadata of the active evidence file currently under inspection
  const [activeEvidence, setActiveEvidence] = useState(null);

  const setActiveView = (newView) => {
    setActiveViewState(newView);
    if (typeof window !== "undefined" && window.location.hash.replace("#", "") !== newView) {
      window.location.hash = newView;
    }
  };

  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace("#", "").toLowerCase();
      if (VALID_VIEWS.includes(hash)) {
        setActiveViewState(hash);
      }
    };
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  const renderView = () => {
    switch (activeView) {
      case "dashboard":
        return (
          <Dashboard 
            setActiveView={setActiveView} 
            setActiveEvidence={setActiveEvidence} 
          />
        );
      case "evidence":
        return (
          <Evidence 
            activeEvidence={activeEvidence} 
            setActiveEvidence={setActiveEvidence} 
          />
        );
      case "analysis":
        return <Analysis activeEvidence={activeEvidence} />;
      case "integrity":
        return <Integrity activeEvidence={activeEvidence} />;
      case "audit":
        return <AuditLog activeEvidence={activeEvidence} />;
      case "reports":
        return <Reports activeEvidence={activeEvidence} />;
      default:
        return (
          <Dashboard 
            setActiveView={setActiveView} 
            setActiveEvidence={setActiveEvidence} 
          />
        );
    }
  };

  return (
    <MainLayout 
      activeView={activeView} 
      setActiveView={setActiveView}
      activeEvidence={activeEvidence}
      setActiveEvidence={setActiveEvidence}
    >
      {renderView()}
    </MainLayout>
  );
}

export default App;
