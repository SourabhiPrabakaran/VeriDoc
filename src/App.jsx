import React, { useState } from "react";
import MainLayout from "./components/layout/MainLayout";
import Dashboard from "./pages/Dashboard";
import Evidence from "./pages/Evidence";
import Analysis from "./pages/Analysis";
import Integrity from "./pages/Integrity";
import AuditLog from "./pages/AuditLog";
import Reports from "./pages/Reports";

function App() {
  const [activeView, setActiveView] = useState("dashboard");
  // Holds metadata of the active evidence file currently under inspection
  const [activeEvidence, setActiveEvidence] = useState(null);

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
