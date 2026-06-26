import React, { useState, useRef, useEffect } from 'react';

const ImageSlider = ({ 
  irImage, 
  colorizedImage, 
  detectionsIr = [], 
  detectionsRgb = [], 
  showDetection = false 
}) => {
  const [sliderPosition, setSliderPosition] = useState(50); // percentage (0-100)
  const isDragging = useRef(false);
  const containerRef = useRef(null);

  // Handle Drag / Move Events
  const handleMove = (clientX) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    let percentage = (x / rect.width) * 100;
    
    // Constraints
    if (percentage < 0) percentage = 0;
    if (percentage > 100) percentage = 100;
    
    setSliderPosition(percentage);
  };

  const handleTouchMove = (e) => {
    if (!isDragging.current) return;
    if (e.touches.length > 0) {
      handleMove(e.touches[0].clientX);
    }
  };

  const handleMouseMove = (e) => {
    if (!isDragging.current) return;
    handleMove(e.clientX);
  };

  const stopDragging = () => {
    isDragging.current = false;
  };

  useEffect(() => {
    // Listen globally to mousemove & mouseup for a smooth drag experience
    const handleGlobalMouseMove = (e) => {
      if (isDragging.current) {
        handleMove(e.clientX);
      }
    };
    const handleGlobalMouseUp = () => {
      isDragging.current = false;
    };

    window.addEventListener('mousemove', handleGlobalMouseMove);
    window.addEventListener('mouseup', handleGlobalMouseUp);
    
    return () => {
      window.removeEventListener('mousemove', handleGlobalMouseMove);
      window.removeEventListener('mouseup', handleGlobalMouseUp);
    };
  }, []);

  const startDragging = (e) => {
    e.preventDefault();
    isDragging.current = true;
  };

  // Helper to draw bounding box divs based on percentage coordinates (assuming 256x256 original size)
  const renderDetections = (detections, isRgb) => {
    const colors = {
      vehicle: '#00f2fe',  // Cyan
      building: '#ff007f', // Magenta
      road: '#00ff00'      // Green
    };
    
    const size = 256; // Grid dimensions returned by API
    
    return detections.map((det, index) => {
      const [x1, y1, x2, y2] = det.box;
      
      // Calculate percentages
      const left = (x1 / size) * 100;
      const top = (y1 / size) * 100;
      const width = ((x2 - x1) / size) * 100;
      const height = ((y2 - y1) / size) * 100;
      
      const borderColor = colors[det.class] || '#ffffff';
      
      return (
        <div 
          key={index}
          style={{
            position: 'absolute',
            left: `${left}%`,
            top: `${top}%`,
            width: `${width}%`,
            height: `${height}%`,
            border: `2px solid ${borderColor}`,
            boxShadow: `0 0 6px ${borderColor}`,
            pointerEvents: 'auto',
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'flex-start'
          }}
          title={`${det.class}: ${Math.round(det.confidence * 100)}%`}
        >
          <span style={{
            background: borderColor,
            color: '#000000',
            fontSize: '9px',
            fontWeight: 'bold',
            padding: '1px 3px',
            fontFamily: 'monospace',
            textTransform: 'uppercase'
          }}>
            {det.class.substring(0, 3)} {Math.round(det.confidence * 100)}%
          </span>
        </div>
      );
    });
  };

  return (
    <div 
      className="slider-wrapper" 
      ref={containerRef}
      onTouchMove={handleTouchMove}
      onTouchStart={() => { isDragging.current = true; }}
      onTouchEnd={stopDragging}
    >
      {/* 1. Behind Layer: Original Monochrome IR */}
      <div 
        className="slider-image slider-before"
        style={{
          width: '100%',
          height: '100%'
        }}
      >
        <img 
          src={`data:image/png;base64,${irImage}`} 
          alt="Original Infrared" 
          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
        />
        
        {/* Detection Boxes on IR (Clipped: only show to the left of the slider) */}
        {showDetection && (
          <div 
            className="bbox-overlay"
            style={{
              clipPath: `polygon(0 0, ${sliderPosition}% 0, ${sliderPosition}% 100%, 0 100%)`
            }}
          >
            {renderDetections(detectionsIr, false)}
          </div>
        )}
      </div>

      {/* 2. Front Layer: Colorized RGB (Clipped) */}
      <div 
        className="slider-image slider-after"
        style={{
          width: '100%',
          height: '100%',
          '--clip-pos': `${sliderPosition}%`
        }}
      >
        <img 
          src={`data:image/png;base64,${colorizedImage}`} 
          alt="Colorized RGB" 
          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
        />
        
        {/* Detection Boxes on RGB (Clipped: only show to the right of the slider) */}
        {showDetection && (
          <div 
            className="bbox-overlay"
            style={{
              clipPath: `polygon(${sliderPosition}% 0, 100% 0, 100% 100%, ${sliderPosition}% 100%)`
            }}
          >
            {renderDetections(detectionsRgb, true)}
          </div>
        )}
      </div>

      {/* Slider Controls Labels */}
      <span className="slider-label left">Mono IR</span>
      <span className="slider-label right">Enhanced RGB</span>

      {/* 3. Drag Handle Line */}
      <div 
        className="slider-handle"
        style={{ '--clip-pos': `${sliderPosition}%` }}
        onMouseDown={startDragging}
      >
        <div className="slider-handle-button">
          {/* SVG Double Arrow Icon */}
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M7 17l-5-5 5-5M17 17l5-5-5-5" />
          </svg>
        </div>
      </div>
    </div>
  );
};

export default ImageSlider;
