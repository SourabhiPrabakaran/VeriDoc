/**
 * Veridoc Forensics - API Service (Phase 5 Integration)
 * 
 * Connecting directly to the local FastAPI backend.
 * Default URL: http://127.0.0.1:8000
 */

const BASE_URL = "http://127.0.0.1:8000/api";

export const apiService = {
  /**
   * GET /api/health
   * Performs a real health check on the backend FastAPI server.
   */
  async healthCheck() {
    try {
      const response = await fetch(`${BASE_URL}/health`, {
        method: "GET",
        headers: { "Accept": "application/json" }
      });
      
      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }
      
      const data = await response.json();
      return {
        status: "ONLINE",
        service: data.service,
        backendConnected: true,
        message: "Forensic backend is reachable and fully functional."
      };
    } catch (err) {
      return {
        status: "OFFLINE",
        service: "veridoc-backend",
        backendConnected: false,
        message: "FastAPI server is unreachable. Verify Uvicorn is running on port 8000."
      };
    }
  },

  /**
   * POST /api/evidence/upload
   * Submits files as multipart/form-data. Calculates SHA-256 on the backend.
   */
  async uploadEvidence(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${BASE_URL}/evidence/upload`, {
      method: "POST",
      body: formData
    });

    if (response.status === 409) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || "Evidence with this SHA-256 hash already exists.");
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Upload request failed with status: ${response.status}`);
    }

    return await response.json();
  },

  /**
   * GET /api/evidence
   * Retrieves list of all registered evidence items.
   */
  async getEvidence() {
    const response = await fetch(`${BASE_URL}/evidence`);
    if (!response.ok) {
      throw new Error(`Failed to list evidence logs: ${response.status}`);
    }
    const data = await response.json();
    return data.items || [];
  },

  /**
   * GET /api/evidence/{evidence_id}
   * Retrieves metadata of a single registered file.
   */
  async getEvidenceById(evidenceId) {
    const response = await fetch(`${BASE_URL}/evidence/${evidenceId}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch evidence metadata for ID ${evidenceId}: ${response.status}`);
    }
    return await response.json();
  },

  /**
   * GET /api/evidence/{evidence_id}/audit
   * Retrieves custody logs matching an evidence ID.
   */
  async getAuditLog(evidenceId) {
    const response = await fetch(`${BASE_URL}/evidence/${evidenceId}/audit`);
    if (!response.ok) {
      throw new Error(`Failed to fetch audit trails for ID ${evidenceId}: ${response.status}`);
    }
    return await response.json();
  },

  /**
   * POST /api/evidence/{evidence_id}/preprocess
   * Triggers the media stream verification, verification check, and face localization pipeline.
   */
  async startPreprocessing(evidenceId) {
    const response = await fetch(`${BASE_URL}/evidence/${evidenceId}/preprocess`, {
      method: "POST"
    });

    if (response.status === 409) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || "SHA-256 integrity check failed. Analysis aborted.");
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Preprocessing failed with status ${response.status}`);
    }

    return await response.json();
  },

  /**
   * GET /api/evidence/{evidence_id}/preprocessing
   * Retrieves the latest face detection results and metadata analysis.
   */
  async getPreprocessing(evidenceId) {
    const response = await fetch(`${BASE_URL}/evidence/${evidenceId}/preprocessing`);
    if (response.status === 404) {
      return null;
    }
    if (!response.ok) {
      throw new Error(`Failed to fetch preprocessing results: ${response.status}`);
    }
    return await response.json();
  },

  /**
   * POST /api/evidence/{evidence_id}/deepfake/analyze
   * Triggers the Deepfake classification pipeline.
   */
  async startDeepfakeAnalysis(evidenceId) {
    const response = await fetch(`${BASE_URL}/evidence/${evidenceId}/deepfake/analyze`, {
      method: "POST"
    });

    if (response.status === 409) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || "Integrity verification failed before analysis.");
    }

    if (response.status === 400) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || "Evidence preprocessing is not completed.");
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Deepfake analysis failed with status: ${response.status}`);
    }

    return await response.json();
  },

  /**
   * GET /api/evidence/{evidence_id}/deepfake
   * Retrieves the latest Deepfake analysis results (labels, confidence, model info).
   */
  async getDeepfakeResults(evidenceId) {
    const response = await fetch(`${BASE_URL}/evidence/${evidenceId}/deepfake`);
    if (response.status === 404) {
      return null;
    }
    if (!response.ok) {
      throw new Error(`Failed to fetch deepfake results: ${response.status}`);
    }
    return await response.json();
  },

  /**
   * POST /api/evidence/{evidence_id}/ela/analyze
   * Triggers Error Level Analysis.
   */
  async startElaAnalysis(evidenceId) {
    const response = await fetch(`${BASE_URL}/evidence/${evidenceId}/ela/analyze`, {
      method: "POST"
    });

    if (response.status === 409) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || "Integrity verification failed before ELA analysis.");
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `ELA failed with status: ${response.status}`);
    }

    return await response.json();
  },

  /**
   * GET /api/evidence/{evidence_id}/ela
   * Retrieves ELA analysis details.
   */
  async getElaResults(evidenceId) {
    const response = await fetch(`${BASE_URL}/evidence/${evidenceId}/ela`);
    if (response.status === 404) {
      return null;
    }
    if (!response.ok) {
      throw new Error(`Failed to fetch ELA results: ${response.status}`);
    }
    return await response.json();
  },

  /**
   * POST /api/evidence/{evidence_id}/fft/analyze
   * Triggers 2D Fast Fourier Transform.
   */
  async startFftAnalysis(evidenceId) {
    const response = await fetch(`${BASE_URL}/evidence/{evidence_id}/fft/analyze`
      .replace("{evidence_id}", evidenceId.toString()), {
      method: "POST"
    });

    if (response.status === 409) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || "Integrity verification failed before FFT analysis.");
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `FFT failed with status: ${response.status}`);
    }

    return await response.json();
  },

  /**
   * GET /api/evidence/{evidence_id}/fft
   * Retrieves 2D FFT analysis details.
   */
  async getFftResults(evidenceId) {
    const response = await fetch(`${BASE_URL}/evidence/${evidenceId}/fft`);
    if (response.status === 404) {
      return null;
    }
    if (!response.ok) {
      throw new Error(`Failed to fetch FFT results: ${response.status}`);
    }
    return await response.json();
  },

  /**
   * Returns the secure controlled endpoint URL to stream original evidence previews.
   */
  getEvidencePreviewUrl(evidenceId) {
    return `${BASE_URL}/evidence/${evidenceId}/preview`;
  },

  /**
   * Returns the secure controlled endpoint URL to serve cropped face regions.
   */
  getFaceCropUrl(cropFilename) {
    return `${BASE_URL}/analysis/crop/${cropFilename}`;
  },

  /**
   * Returns the secure controlled endpoint URL to serve ELA visualizations.
   */
  getElaImageUrl(filename) {
    return `${BASE_URL}/analysis/ela/${filename}`;
  },

  /**
   * Returns the secure controlled endpoint URL to serve FFT visualizations.
   */
  getFftImageUrl(filename) {
    return `${BASE_URL}/analysis/fft/${filename}`;
  },

  /**
   * POST /api/evidence/{evidence_id}/report
   * Triggers forensic report generation for the evidence item.
   */
  async generateReport(evidenceId) {
    const response = await fetch(`${BASE_URL}/evidence/${evidenceId}/report`, {
      method: "POST"
    });

    if (response.status === 404) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || "Evidence item not found.");
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Report generation failed with status: ${response.status}`);
    }

    return await response.json();
  },

  /**
   * GET /api/evidence/{evidence_id}/reports
   * Retrieves a list of reports generated for the evidence item.
   */
  async getReports(evidenceId) {
    const response = await fetch(`${BASE_URL}/evidence/${evidenceId}/reports`);
    if (!response.ok) {
      throw new Error(`Failed to fetch report list: ${response.status}`);
    }
    const data = await response.json();
    return data.items || [];
  },

  /**
   * Returns the secure controlled endpoint URL to download the report PDF.
   */
  getReportDownloadUrl(reportUuid) {
    return `${BASE_URL}/reports/${reportUuid}/download`;
  }
};
