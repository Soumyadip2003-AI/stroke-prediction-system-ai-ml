import React, { useState, useEffect } from 'react';
import './App.css';

// Components
import Navigation from './components/Navigation';
import Hero from './components/Hero';
import Assessment from './components/Assessment';
import Results from './components/Results';
import Insights from './components/Insights';
import About from './components/About';
import Footer from './components/Footer';
import LoadingOverlay from './components/LoadingOverlay';

function App() {
  const [currentSection, setCurrentSection] = useState('home');
  const [assessmentData, setAssessmentData] = useState(null);
  const [isLoading] = useState(false);

  // Initialize particles.js background
  useEffect(() => {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js';
    script.onload = () => {
      if ((window as any).particlesJS) {
        (window as any).particlesJS('particles-js', {
          particles: {
            number: { value: 80 },
            color: { value: '#667eea' },
            shape: { type: 'circle' },
            opacity: { value: 0.5 },
            size: { value: 3 },
            line_linked: {
              enable: true,
              distance: 150,
              color: '#667eea',
              opacity: 0.4,
              width: 1
            },
            move: {
              enable: true,
              speed: 2,
              direction: 'none',
              random: false,
              straight: false,
              out_mode: 'out',
              bounce: false
            }
          },
          interactivity: {
            detect_on: 'canvas',
            events: {
              onhover: { enable: true, mode: 'repulse' },
              onclick: { enable: true, mode: 'push' },
              resize: true
            }
          },
          retina_detect: true
        });
      }
    };
    document.head.appendChild(script);

    return () => {
      document.head.removeChild(script);
    };
  }, []);

  const handleAssessmentComplete = (data: any) => {
    setAssessmentData(data);
    setCurrentSection('results');
  };

  const scrollToSection = (sectionId: string) => {
    setCurrentSection(sectionId);
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="App bg-gray-900 text-white min-h-screen">
      {/* Animated Background */}
      <div id="particles-js" className="fixed inset-0 z-0"></div>
      
      {/* Navigation */}
      <Navigation currentSection={currentSection} onNavigate={scrollToSection} />
      
      {/* Hero Section */}
      <section id="home" className="min-h-screen flex items-center justify-center relative">
        <Hero onStartAssessment={() => scrollToSection('assessment')} />
      </section>
      
      {/* Assessment Section */}
      <section id="assessment" className="py-20 bg-gray-800">
        <Assessment onComplete={handleAssessmentComplete} />
      </section>
      
      {/* Results Section */}
      {assessmentData && (
        <section id="results" className="py-20 bg-gray-900">
          <Results data={assessmentData} />
        </section>
      )}
      
      {/* Insights Section */}
      <section id="insights" className="py-20 bg-gray-800">
        <Insights />
      </section>
      
      {/* About Section */}
      <section id="about" className="py-20 bg-gray-900">
        <About />
      </section>
      
      {/* Footer */}
      <Footer />
      
      {/* Loading Overlay */}
      {isLoading && <LoadingOverlay />}
    </div>
  );
}

export default App;
