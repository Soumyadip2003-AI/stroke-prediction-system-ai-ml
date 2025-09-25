import React, { useEffect, useRef, useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBrain, faRobot, faMagicWandSparkles } from '@fortawesome/free-solid-svg-icons';

interface HeroProps {
  onStartAssessment: () => void;
}

const Hero: React.FC<HeroProps> = ({ onStartAssessment }) => {
  const networkRef = useRef<HTMLDivElement>(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isHovering, setIsHovering] = useState(false);

  // Enhanced mouse tracking for neural network interactions
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (networkRef.current) {
        const rect = networkRef.current.getBoundingClientRect();
        setMousePosition({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top
        });
      }
    };

    const handleMouseEnter = () => {
      setIsHovering(true);
      // Add a special effect when entering
      if (networkRef.current) {
        networkRef.current.style.animation = 'neural-network-pulse 0.5s ease-out';
      }
    };

    const handleMouseLeave = () => {
      setIsHovering(false);
      // Reset animation when leaving
      if (networkRef.current) {
        networkRef.current.style.animation = '';
      }
    };

    const handleClick = (e: MouseEvent) => {
      // Add click ripple effect
      const ripple = document.createElement('div');
      ripple.className = 'click-ripple-effect';
      ripple.style.left = `${e.offsetX}px`;
      ripple.style.top = `${e.offsetY}px`;

      if (networkRef.current) {
        networkRef.current.appendChild(ripple);
        setTimeout(() => {
          ripple.remove();
        }, 600);
      }
    };

    const network = networkRef.current;
    if (network) {
      network.addEventListener('mousemove', handleMouseMove);
      network.addEventListener('mouseenter', handleMouseEnter);
      network.addEventListener('mouseleave', handleMouseLeave);
      network.addEventListener('click', handleClick);
    }

    // Add keyboard interaction
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === 'Enter' || e.key === ' ') {
        // Trigger a neural network burst effect
        if (network) {
          network.style.animation = 'neural-network-burst 0.8s ease-out';
          setTimeout(() => {
            network.style.animation = '';
          }, 800);
        }
      }
    };

    document.addEventListener('keypress', handleKeyPress);

    return () => {
      if (network) {
        network.removeEventListener('mousemove', handleMouseMove);
        network.removeEventListener('mouseenter', handleMouseEnter);
        network.removeEventListener('mouseleave', handleMouseLeave);
        network.removeEventListener('click', handleClick);
      }
      document.removeEventListener('keypress', handleKeyPress);
    };
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 lg:py-20">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 items-center">
        <div className="space-y-4 sm:space-y-6 lg:space-y-8 text-center lg:text-left">
          <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl xl:text-7xl font-bold leading-tight">
            <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              Advanced AI
            </span>
            <br />
            <span className="text-white">Stroke Risk Prediction</span>
          </h1>
          <p className="text-base sm:text-lg lg:text-xl text-gray-300 leading-relaxed max-w-2xl mx-auto lg:mx-0">
            Experience advanced healthcare AI using supervised and unsupervised learning with 6 models.
            Get personalized stroke risk predictions with 88.56% accuracy and intelligent recommendations.
          </p>

          {/* AI Indicators */}
          <div className="flex flex-wrap items-center justify-center lg:justify-start gap-2 sm:gap-4 mb-4 lg:mb-6">
            <div className="flex items-center bg-blue-900 bg-opacity-50 px-2 sm:px-3 lg:px-4 py-1 sm:py-2 rounded-full interactive-card">
              <FontAwesomeIcon icon={faBrain} className="text-blue-400 mr-1 sm:mr-2 pulse-on-hover text-xs sm:text-sm" />
              <span className="text-blue-400 text-xs sm:text-sm font-medium interactive-text">Advanced ML Models</span>
            </div>
            <div className="flex items-center bg-green-900 bg-opacity-50 px-2 sm:px-3 lg:px-4 py-1 sm:py-2 rounded-full interactive-card">
              <span className="text-green-400 text-xs sm:text-sm font-medium interactive-text">High Accuracy</span>
            </div>
            <div className="flex items-center bg-purple-900 bg-opacity-50 px-2 sm:px-3 lg:px-4 py-1 sm:py-2 rounded-full interactive-card">
              <FontAwesomeIcon icon={faMagicWandSparkles} className="text-purple-400 mr-1 sm:mr-2 pulse-on-hover text-xs sm:text-sm" />
              <span className="text-purple-400 text-xs sm:text-sm font-medium interactive-text">Intelligent Analysis</span>
            </div>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4 lg:gap-6">
            <div className="text-center interactive-card bounce-on-hover p-2 sm:p-3">
              <div className="text-lg sm:text-2xl lg:text-3xl font-bold text-blue-400 interactive-text">88.56%</div>
              <div className="text-xs sm:text-sm text-gray-400">Accuracy</div>
            </div>
            <div className="text-center interactive-card bounce-on-hover p-2 sm:p-3">
              <div className="text-lg sm:text-2xl lg:text-3xl font-bold text-purple-400 interactive-text">6</div>
              <div className="text-xs sm:text-sm text-gray-400">AI Models</div>
            </div>
            <div className="text-center interactive-card bounce-on-hover p-2 sm:p-3">
              <div className="text-lg sm:text-2xl lg:text-3xl font-bold text-green-400 interactive-text">∞</div>
              <div className="text-xs sm:text-sm text-gray-400">Learning</div>
            </div>
            <div className="text-center interactive-card bounce-on-hover p-2 sm:p-3">
              <div className="text-lg sm:text-2xl lg:text-3xl font-bold text-pink-400 interactive-text">20+</div>
              <div className="text-xs sm:text-sm text-gray-400">Features</div>
            </div>
          </div>
          <button
            onClick={onStartAssessment}
            className="magic-button text-white px-4 sm:px-6 lg:px-8 py-3 sm:py-4 rounded-full font-semibold transition-all duration-300 transform click-ripple text-sm sm:text-base w-full sm:w-auto min-w-[200px]"
          >
            <FontAwesomeIcon icon={faRobot} className="mr-2 animate-spin hover:animate-none" />
            <span className="relative z-10">🚀 Start Interactive AI Assessment</span>
          </button>
        </div>
        <div className="flex justify-center mt-8 lg:mt-0">
          <div
            ref={networkRef}
            id="neural-network-container"
            className="neural-network-container relative cursor-pointer enhanced-particles w-64 h-64 sm:w-80 sm:h-80 lg:w-96 lg:h-96"
            style={{ touchAction: 'none' }} // Prevent default touch behaviors
          >
            {/* Neural Network Connections */}
            <svg className="neural-connections absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 200 200">
              <defs>
                <linearGradient id="connectionGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#667eea" stopOpacity="0.3" />
                  <stop offset="50%" stopColor="#764ba2" stopOpacity="0.6" />
                  <stop offset="100%" stopColor="#f093fb" stopOpacity="0.3" />
                </linearGradient>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                  <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                  </feMerge>
                </filter>
              </defs>

              {/* Dynamic connections based on mouse position */}
              <line x1="20" y1="20" x2="180" y2="20" stroke="url(#connectionGradient)" strokeWidth="1" opacity="0.4" filter="url(#glow)" />
              <line x1="20" y1="20" x2="100" y2="100" stroke="url(#connectionGradient)" strokeWidth="1" opacity="0.4" filter="url(#glow)" />
              <line x1="180" y1="20" x2="100" y2="100" stroke="url(#connectionGradient)" strokeWidth="1" opacity="0.4" filter="url(#glow)" />
              <line x1="20" y1="180" x2="100" y2="100" stroke="url(#connectionGradient)" strokeWidth="1" opacity="0.4" filter="url(#glow)" />
              <line x1="180" y1="180" x2="100" y2="100" stroke="url(#connectionGradient)" strokeWidth="1" opacity="0.4" filter="url(#glow)" />

              {/* Mouse interaction lines */}
              {isHovering && (
                <>
                  <line x1={mousePosition.x} y1={mousePosition.y} x2="20" y2="20" stroke="#f093fb" strokeWidth="2" opacity="0.8" filter="url(#glow)" />
                  <line x1={mousePosition.x} y1={mousePosition.y} x2="180" y2="20" stroke="#667eea" strokeWidth="2" opacity="0.8" filter="url(#glow)" />
                  <line x1={mousePosition.x} y1={mousePosition.y} x2="100" y2="100" stroke="#764ba2" strokeWidth="2" opacity="0.8" filter="url(#glow)" />
                  <line x1={mousePosition.x} y1={mousePosition.y} x2="20" y2="180" stroke="#f5576c" strokeWidth="2" opacity="0.8" filter="url(#glow)" />
                  <line x1={mousePosition.x} y1={mousePosition.y} x2="180" y2="180" stroke="#4facfe" strokeWidth="2" opacity="0.8" filter="url(#glow)" />
                </>
              )}
            </svg>

            {/* Interactive Neural Network */}
            <div className="neural-network">
              <div className="neuron" data-id="1" style={{ '--mouse-x': mousePosition.x, '--mouse-y': mousePosition.y } as any}></div>
              <div className="neuron" data-id="2" style={{ '--mouse-x': mousePosition.x, '--mouse-y': mousePosition.y } as any}></div>
              <div className="neuron" data-id="3" style={{ '--mouse-x': mousePosition.x, '--mouse-y': mousePosition.y } as any}></div>
              <div className="neuron" data-id="4" style={{ '--mouse-x': mousePosition.x, '--mouse-y': mousePosition.y } as any}></div>
              <div className="neuron" data-id="5" style={{ '--mouse-x': mousePosition.x, '--mouse-y': mousePosition.y } as any}></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hero;
