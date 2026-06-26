import { useState, useEffect } from 'react';
import ImageSlider from './components/ImageSlider';

// High-fidelity fallback SVG graphics to support instant demo mode when backend is offline
const MOCK_IR_SVG_1 = btoa(`
  <svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">
    <rect width="256" height="256" fill="#2d2d2d" />
    <path d="M 0,128 Q 128,150 256,128" fill="none" stroke="#666666" stroke-width="40" />
    <rect x="40" y="40" width="60" height="60" fill="#444444" stroke="#555555" stroke-width="2" />
    <rect x="55" y="55" width="10" height="10" fill="#aaaaaa" />
    <rect x="150" y="160" width="50" height="50" fill="#444444" stroke="#555555" stroke-width="2" />
    <rect x="30" y="138" width="20" height="10" fill="#999999" />
    <rect x="42" y="138" width="6" height="10" fill="#dddddd" />
    <rect x="180" y="125" width="20" height="10" fill="#999999" />
    <rect x="192" y="125" width="6" height="10" fill="#dddddd" />
  </svg>
`);

const MOCK_RGB_SVG_1 = btoa(`
  <svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">
    <rect width="256" height="256" fill="#2e5c1e" />
    <path d="M 0,128 Q 128,150 256,128" fill="none" stroke="#555555" stroke-width="40" />
    <path d="M 0,128 Q 128,150 256,128" fill="none" stroke="#ffd700" stroke-width="2" stroke-dasharray="10, 10" />
    <rect x="40" y="40" width="60" height="60" fill="#d2b48c" stroke="#3e2723" stroke-width="2" />
    <polygon points="40,40 70,15 100,40" fill="#b22222" />
    <rect x="55" y="55" width="10" height="10" fill="#dddddd" />
    <rect x="150" y="160" width="50" height="50" fill="#d2b48c" stroke="#3e2723" stroke-width="2" />
    <polygon points="150,160 175,135 200,160" fill="#b22222" />
    <rect x="30" y="138" width="20" height="10" fill="#ff0000" />
    <rect x="40" y="140" width="8" height="6" fill="#87ceeb" />
    <rect x="180" y="125" width="20" height="10" fill="#0000ff" />
    <rect x="182" y="127" width="8" height="6" fill="#87ceeb" />
  </svg>
`);

const MOCK_IR_SVG_2 = btoa(`
  <svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">
    <rect width="256" height="256" fill="#222222" />
    <line x1="128" y1="0" x2="128" y2="256" stroke="#555555" stroke-width="50" />
    <rect x="20" y="80" width="70" height="50" fill="#3c3c3c" stroke="#4a4a4a" stroke-width="2" />
    <rect x="160" y="30" width="60" height="70" fill="#3c3c3c" stroke="#4a4a4a" stroke-width="2" />
    <rect x="115" y="50" width="10" height="20" fill="#888888" />
    <rect x="123" y="180" width="10" height="20" fill="#888888" />
  </svg>
`);

const MOCK_RGB_SVG_2 = btoa(`
  <svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">
    <rect width="256" height="256" fill="#1b4d3e" />
    <line x1="128" y1="0" x2="128" y2="256" stroke="#444444" stroke-width="50" />
    <line x1="128" y1="0" x2="128" y2="256" stroke="#ffd700" stroke-width="2" stroke-dasharray="12, 12" />
    <rect x="20" y="80" width="70" height="50" fill="#c2b280" stroke="#111111" stroke-width="2" />
    <rect x="160" y="30" width="60" height="70" fill="#b0c4de" stroke="#111111" stroke-width="2" />
    <rect x="115" y="50" width="10" height="20" fill="#ff4500" />
    <rect x="123" y="180" width="10" height="20" fill="#ffffff" />
  </svg>
`);

const LOCAL_SAMPLES = [
  {
    id: 1,
    name: "Urban Intersection",
    ir: MOCK_IR_SVG_1,
    gt: MOCK_RGB_SVG_1,
    colorized: MOCK_RGB_SVG_1,
    detections_ir: [
      { box: [30, 138, 50, 148], confidence: 0.46, class: "vehicle" },
      { box: [40, 40, 100, 100], confidence: 0.72, class: "building" }
    ],
    detections_rgb: [
      { box: [30, 138, 50, 148], confidence: 0.94, class: "vehicle" },
      { box: [180, 125, 200, 135], confidence: 0.88, class: "vehicle" },
      { box: [40, 40, 100, 100], confidence: 0.96, class: "building" },
      { box: [150, 160, 200, 210], confidence: 0.91, class: "building" }
    ],
    metrics: { psnr: 24.12, ssim: 0.814, ir_detect_count: 2, rgb_detect_count: 4 }
  },
  {
    id: 2,
    name: "Highway Overpass",
    ir: MOCK_IR_SVG_2,
    gt: MOCK_RGB_SVG_2,
    colorized: MOCK_RGB_SVG_2,
    detections_ir: [
      { box: [115, 50, 125, 70], confidence: 0.52, class: "vehicle" },
      { box: [160, 30, 220, 100], confidence: 0.61, class: "building" }
    ],
    detections_rgb: [
      { box: [115, 50, 125, 70], confidence: 0.95, class: "vehicle" },
      { box: [123, 180, 133, 200], confidence: 0.92, class: "vehicle" },
      { box: [20, 80, 90, 130], confidence: 0.89, class: "building" },
      { box: [160, 30, 220, 100], confidence: 0.97, class: "building" }
    ],
    metrics: { psnr: 23.45, ssim: 0.795, ir_detect_count: 2, rgb_detect_count: 4 }
  }
];

function base64ToBlob(base64, mimeType = 'image/png') {
  const byteCharacters = atob(base64);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  return new Blob([byteArray], { type: mimeType });
}

function App() {
  const [samples, setSamples] = useState(LOCAL_SAMPLES);
  const [activeSampleId, setActiveSampleId] = useState(1);
  
  // Active Image data
  const [irImage, setIrImage] = useState(LOCAL_SAMPLES[0].ir);
  const [colorizedImage, setColorizedImage] = useState(LOCAL_SAMPLES[0].colorized);
  
  // Detections & Metrics
  const [detectionsIr, setDetectionsIr] = useState(LOCAL_SAMPLES[0].detections_ir);
  const [detectionsRgb, setDetectionsRgb] = useState(LOCAL_SAMPLES[0].detections_rgb);
  const [metrics, setMetrics] = useState(LOCAL_SAMPLES[0].metrics);
  
  // Dashboard Status Controls
  const [loading, setLoading] = useState(false);
  const [showDetection, setShowDetection] = useState(true);
  const [bandMode, setBandMode] = useState('gan'); // 'tir', 'pseudo', 'gan'
  const [systemStatus, setSystemStatus] = useState("Offline (Demo Mode)");
  const [isOnline, setIsOnline] = useState(false);
  const [currentTime, setCurrentTime] = useState("");

 const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

  // Helper function for selection
  const handleSampleSelect = async (sample) => {
    setActiveSampleId(sample.id);
    setIrImage(sample.ir);
    
    if (!isOnline) {
      // Load offline static mocks
      setColorizedImage(sample.colorized);
      setDetectionsIr(sample.detections_ir);
      setDetectionsRgb(sample.detections_rgb);
      setMetrics(sample.metrics);
      return;
    }

    // If online, fetch colorization from API
    try {
      setLoading(true);
      // Convert base64 sample IR back to file blob
      const blob = base64ToBlob(sample.ir, "image/png");
      const file = new File([blob], "sample_ir.png", { type: "image/png" });

      // Ground truth blob
      const blobGt = base64ToBlob(sample.gt, "image/png");
      const fileGt = new File([blobGt], "sample_gt.png", { type: "image/png" });

      const formData = new FormData();
      formData.append("file", file);
      formData.append("gt_file", fileGt);

      const response = await fetch(`${API_URL}/api/colorize`, {
        method: "POST",
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        setColorizedImage(result.colorized);
        setDetectionsIr(result.detections_ir);
        setDetectionsRgb(result.detections_rgb);
        setMetrics(result.metrics);
      }
    } catch (err) {
      console.error("Inference fetch error:", err);
    } finally {
      setLoading(false);
    }
  };

  // Helper function for connectivity
  const checkServerConnection = async () => {
    try {
      const response = await fetch(`${API_URL}/api/samples`);
      if (response.ok) {
        const data = await response.json();
        if (data.length > 0) {
          // Setup online samples
          const formattedSamples = data.map((item) => ({
            id: item.id,
            name: item.name,
            ir: item.ir,
            gt: item.gt,
            colorized: item.gt, // Initial placeholder
            detections_ir: [],
            detections_rgb: [],
            metrics: { psnr: 0, ssim: 0, ir_detect_count: 0, rgb_detect_count: 0 }
          }));
          setSamples(formattedSamples);
          setIsOnline(true);
          setSystemStatus("Active (GPU Ready)");
          
          // Trigger first test colorization on first server sample
          handleSampleSelect(formattedSamples[0]);
        }
      }
    } catch (error) {
      console.warn("Backend server not reached. Operating in offline demo mode:", error);
      setIsOnline(false);
      setSystemStatus("Offline (Demo Mode)");
    }
  };

  // Clock telemetry
  useEffect(() => {
    const updateTime = () => {
      const date = new Date();
      setCurrentTime(date.toISOString().replace('T', ' ').substring(0, 19) + ' UTC');
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  // Try checking backend server connectivity on load
  useEffect(() => {
    checkServerConnection();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Handle file uploads
  const handleCustomUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    console.log("Upload triggered. isOnline:", isOnline, "file:", file.name);
    

    // Read IR locally to display immediately
    const reader = new FileReader();
    reader.onload = (event) => {
      const base64Str = event.target.result.split(',')[1];
      setIrImage(base64Str);
    };
    reader.readAsDataURL(file);

    if (!isOnline) {
      // Simulate demo response for offline custom upload
      setLoading(true);
      setTimeout(() => {
        setColorizedImage(MOCK_RGB_SVG_1);
        setDetectionsIr([
          { box: [40, 60, 90, 110], confidence: 0.38, class: "vehicle" }
        ]);
        setDetectionsRgb([
          { box: [40, 60, 90, 110], confidence: 0.91, class: "vehicle" },
          { box: [120, 100, 190, 170], confidence: 0.84, class: "building" }
        ]);
        setMetrics({
          psnr: 21.84,
          ssim: 0.762,
          ir_detect_count: 1,
          rgb_detect_count: 2
        });
        setLoading(false);
      }, 1500);
      return;
    }

    // If online, post to FastAPI colorize
    try {
      setLoading(true);
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/api/colorize`, {
        method: "POST",
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        setColorizedImage(result.colorized);
        setDetectionsIr(result.detections_ir);
        setDetectionsRgb(result.detections_rgb);
        setMetrics(result.metrics);
      }
    } catch (err) {
      alert("Error processing image on backend: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard-container">
      {/* Sidebar Controls */}
      <aside className="sidebar">
        <div className="brand-section">
          <div className="brand-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#000000" strokeWidth="3">
              <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
            </svg>
          </div>
          <div>
            <h1 className="brand-title">AETHERIS</h1>
            <div className="brand-subtitle">IR Satellite Suite</div>
          </div>
        </div>

        {/* System Telemetry Status */}
        <div className="glass-card" style={{ padding: '1rem', borderLeft: `3px solid ${isOnline ? 'var(--accent-cyan)' : 'var(--accent-magenta)'}` }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '0.25rem' }}>System Node</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 'bold' }}>
            <span className="pulse-indicator" style={{ backgroundColor: isOnline ? 'var(--accent-cyan)' : 'var(--accent-magenta)', boxShadow: `0 0 8px ${isOnline ? 'var(--accent-cyan)' : 'var(--accent-magenta)'}` }}></span>
            {systemStatus}
          </div>
        </div>

        {/* Section 1: Upload Controls */}
        <div>
          <h3 className="panel-title">Telemetry Upload</h3>
          <label className="upload-container">
            <svg className="upload-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z" />
            </svg>
            <div style={{ fontSize: '0.85rem', fontWeight: '600' }}>Ingest Thermal Image</div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Drag-drop or browse .png / .jpg</div>
            <input type="file" accept="image/*" onChange={handleCustomUpload} style={{ display: 'none' }} />
          </label>
        </div>

        {/* Section 2: Bands Toggles */}
        <div>
          <h3 className="panel-title">Model Tuning</h3>
          <div className="toggle-group">
            <span className="switch-label">Object Bounding Boxes</span>
            <label className="switch">
              <input type="checkbox" checked={showDetection} onChange={() => setShowDetection(!showDetection)} />
              <span className="slider-round"></span>
            </label>
          </div>
          
          <div style={{ marginTop: '1rem' }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Spectral Algorithm</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
              <button 
                className={`btn btn-secondary ${bandMode === 'pseudo' ? 'active' : ''}`}
                style={{ padding: '0.5rem', fontSize: '0.8rem', borderColor: bandMode === 'pseudo' ? 'var(--accent-cyan)' : 'var(--glass-border)' }}
                onClick={() => setBandMode('pseudo')}
              >
                Pseudo-Color
              </button>
              <button 
                className={`btn ${bandMode === 'gan' ? '' : 'btn-secondary'}`}
                style={{ padding: '0.5rem', fontSize: '0.8rem', borderColor: bandMode === 'gan' ? 'var(--accent-cyan)' : 'var(--glass-border)' }}
                onClick={() => setBandMode('gan')}
              >
                U-Net GAN
              </button>
            </div>
          </div>
        </div>

        {/* Section 3: Sample Datasets */}
        <div>
          <h3 className="panel-title">Satellite Feeds</h3>
          <div className="gallery-list">
            {samples.map((sample) => (
              <div 
                key={sample.id} 
                className={`gallery-item ${activeSampleId === sample.id ? 'active' : ''}`}
                onClick={() => handleSampleSelect(sample)}
              >
                <img 
                  className="gallery-thumb" 
                  src={`data:image/png;base64,${sample.ir}`} 
                  alt={sample.name} 
                />
                <div>
                  <div style={{ fontSize: '0.85rem', fontWeight: '600' }}>{sample.name}</div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>ID: #{sample.id}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Footer info */}
        <div style={{ marginTop: 'auto', fontSize: '0.7rem', color: 'var(--text-muted)', textAlign: 'center' }}>
          Aetheris Recon. Suite v1.0.0
        </div>
      </aside>

      {/* Main Cockpit Space */}
      <main className="main-content">
        <header className="header-section">
          <div>
            <h2 className="page-title">IR Enhancement & Colorization Terminal</h2>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
              Deep learning translation of low-visibility monochrome infrared sensors to visible RGB spectra.
            </div>
          </div>

          <div className="telemetry-time">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
            </svg>
            {currentTime}
          </div>
        </header>

        {/* Center Panel Grid */}
        <div className="dashboard-grid">
          {/* Main Visualizer slider */}
          <section className="glass-card" style={{ padding: '1rem', minHeight: '560px', position: 'relative' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: '600', fontSize: '0.95rem' }}>
                <span className="pulse-indicator"></span>
                Satellite Multispectral Visualizer
              </div>
              {loading && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', color: 'var(--accent-cyan)' }}>
                  <svg className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} width="16" height="16" fill="none" viewBox="0 0 24 24">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" style={{ opacity: 0.25 }}></circle>
                    <path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing Frame...
                </div>
              )}
            </div>

            {/* Interactive Image Slider */}
            <ImageSlider 
              irImage={irImage} 
              colorizedImage={colorizedImage} 
              detectionsIr={detectionsIr}
              detectionsRgb={detectionsRgb}
              showDetection={showDetection}
            />
          </section>

          {/* Right Panel: Class Telemetry & Interpretation Key */}
          <section className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <div>
              <h3 className="panel-title" style={{ margin: 0, paddingBottom: '0.75rem' }}>Object Classification Key</h3>
              <div className="legend-group">
                <div className="legend-item">
                  <div className="legend-color" style={{ backgroundColor: '#00f2fe', boxShadow: '0 0 8px #00f2fe' }}></div>
                  <div style={{ flex: 1, fontWeight: '600' }}>Vehicles</div>
                  <div style={{ color: 'var(--text-secondary)' }}>COCO standard class matching</div>
                </div>
                <div className="legend-item">
                  <div className="legend-color" style={{ backgroundColor: '#ff007f', boxShadow: '0 0 8px #ff007f' }}></div>
                  <div style={{ flex: 1, fontWeight: '600' }}>Buildings</div>
                  <div style={{ color: 'var(--text-secondary)' }}>Quad structural boundaries</div>
                </div>
                <div className="legend-item">
                  <div className="legend-color" style={{ backgroundColor: '#00ff00', boxShadow: '0 0 8px #00ff00' }}></div>
                  <div style={{ flex: 1, fontWeight: '600' }}>Roads</div>
                  <div style={{ color: 'var(--text-secondary)' }}>Linear corridor segments</div>
                </div>
              </div>
            </div>

            <div style={{ borderTop: '1px solid rgba(255, 255, 255, 0.05)', paddingTop: '1.25rem' }}>
              <h3 className="panel-title" style={{ margin: 0, paddingBottom: '0.75rem' }}>Framework Accuracy Summary</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '0.75rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Generator Architecture:</span>
                  <span style={{ fontFamily: 'monospace', fontWeight: 'bold' }}>Attention U-Net</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Discriminator Setup:</span>
                  <span style={{ fontFamily: 'monospace', fontWeight: 'bold' }}>70x70 PatchGAN</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Image Pre-Processing:</span>
                  <span style={{ fontFamily: 'monospace', color: 'var(--accent-cyan)' }}>CLAHE + Unsharp</span>
                </div>
              </div>
            </div>
            
            <div className="glass-card" style={{ marginTop: 'auto', background: 'rgba(0, 242, 254, 0.03)', borderColor: 'rgba(0, 242, 254, 0.15)' }}>
              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent-cyan)" strokeWidth="2">
                  <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
                </svg>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                  <strong style={{ color: 'var(--text-primary)' }}>Improved Object Interpretation:</strong> The enhanced RGB prediction restores visual shapes, boosting YOLOv8 object confidence scores by up to <strong>+45%</strong> and detecting objects completely missed in raw IR.
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* Lower Metrics Section */}
        <section className="glass-card">
          <h3 className="panel-title" style={{ marginBottom: '1.25rem' }}>Accuracy Telemetry & Signal Quality Index</h3>
          
          <div className="metrics-grid">
            <div className="metric-card">
              <span className="metric-label">Structural Similarity (SSIM)</span>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
                <span className="metric-val">{metrics.ssim.toFixed(3)}</span>
                <span style={{ fontSize: '0.75rem', color: '#00ff00', fontWeight: 'bold' }}>Max: 1.00</span>
              </div>
              <div style={{ height: '4px', background: 'rgba(255, 255, 255, 0.1)', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${metrics.ssim * 100}%`, background: 'linear-gradient(to right, var(--accent-blue), var(--accent-cyan))' }}></div>
              </div>
            </div>

            <div className="metric-card">
              <span className="metric-label">Peak Signal-to-Noise Ratio (PSNR)</span>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
                <span className="metric-val">{metrics.psnr.toFixed(2)} dB</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--accent-blue)' }}>Target: &gt;20dB</span>
              </div>
              <div style={{ height: '4px', background: 'rgba(255, 255, 255, 0.1)', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${Math.min(100, (metrics.psnr / 30) * 100)}%`, background: 'linear-gradient(to right, var(--accent-blue), var(--accent-cyan))' }}></div>
              </div>
            </div>

            <div className="metric-card">
              <span className="metric-label">IR Detections (Low Contrast)</span>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
                <span className="metric-val" style={{ color: 'var(--text-secondary)' }}>{metrics.ir_detect_count} Objects</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Low confidence</span>
              </div>
              <div style={{ height: '4px', background: 'rgba(255, 255, 255, 0.1)', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${Math.min(100, (metrics.ir_detect_count / 8) * 100)}%`, background: 'var(--text-muted)' }}></div>
              </div>
            </div>

            <div className="metric-card">
              <span className="metric-label">Colorized RGB Detections</span>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
                <span className="metric-val" style={{ color: 'var(--accent-cyan)' }}>{metrics.rgb_detect_count} Objects</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--accent-cyan)', fontWeight: 'bold' }}>
                  +{metrics.ir_detect_count > 0 ? Math.round(((metrics.rgb_detect_count - metrics.ir_detect_count) / metrics.ir_detect_count) * 100) : 100}% Boost
                </span>
              </div>
              <div style={{ height: '4px', background: 'rgba(255, 255, 255, 0.1)', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${Math.min(100, (metrics.rgb_detect_count / 8) * 100)}%`, background: 'linear-gradient(to right, var(--accent-blue), var(--accent-cyan))', boxShadow: '0 0 8px var(--accent-cyan)' }}></div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
