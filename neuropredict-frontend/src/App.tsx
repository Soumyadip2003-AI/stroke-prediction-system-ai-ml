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

  // Initialize enhanced interactive particles.js background
  useEffect(() => {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js';
    script.onload = () => {
      if ((window as any).particlesJS) {
        // Detect mobile device and adjust particle count
        const isMobile = window.innerWidth <= 768;
        const particleCount = isMobile ? 80 : 150;
        const particleDensity = isMobile ? 600 : 1000;

        (window as any).particlesJS('particles-js', {
          particles: {
            number: {
              value: particleCount,
              density: {
                enable: true,
                value_area: particleDensity
              }
            },
            color: {
              value: ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#ff6b6b', '#4ecdc4']
            },
            shape: {
              type: ['circle', 'triangle', 'polygon', 'edge', 'star'],
              stroke: {
                width: 0,
                color: '#fff'
              },
              polygon: {
                nb_sides: 5
              },
              image: {
                src: 'https://particles.js.org/images/circle.png',
                width: 100,
                height: 100
              }
            },
            opacity: {
              value: 0.7,
              random: true,
              anim: {
                enable: true,
                speed: 1.5,
                opacity_min: 0.1,
                sync: false
              }
            },
            size: {
              value: 4,
              random: true,
              anim: {
                enable: true,
                speed: 3,
                size_min: 0.5,
                sync: false
              }
            },
            line_linked: {
              enable: true,
              distance: 150,
              color: '#667eea',
              opacity: 0.8,
              width: 2,
              shadow: {
                enable: true,
                blur: 8,
                color: '#667eea'
              }
            },
            move: {
              enable: true,
              speed: 2,
              direction: 'none',
              random: true,
              straight: false,
              out_mode: 'bounce',
              bounce: true,
              attract: {
                enable: true,
                rotateX: 800,
                rotateY: 1600
              }
            }
          },
          interactivity: {
            detect_on: 'canvas',
            events: {
              onhover: {
                enable: true,
                mode: ['grab', 'bubble', 'repulse']
              },
              onclick: {
                enable: true,
                mode: ['push', 'remove', 'bubble', 'repulse']
              },
              ondiv: {
                elementId: 'neural-network-container',
                enable: true,
                mode: ['bubble', 'grab']
              },
              resize: true
            },
            modes: {
              grab: {
                distance: 250,
                line_linked: {
                  opacity: 1,
                  color: '#f093fb'
                }
              },
              bubble: {
                distance: 250,
                size: 8,
                duration: 3,
                opacity: 0.9,
                speed: 4,
                color: '#4facfe'
              },
              repulse: {
                distance: 120,
                duration: 0.6,
                speed: 2
              },
              push: {
                particles_nb: 6,
                color: '#f5576c'
              },
              remove: {
                particles_nb: 3
              }
            }
          },
          retina_detect: true
        });

        // Add advanced mouse tracking and particle interactions
        let mouseX = 0;
        let mouseY = 0;
        let mouseSpeed = 0;
        let lastMouseX = 0;
        let lastMouseY = 0;
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        let clickCount = 0;
        const canvas = document.querySelector('#particles-js canvas');

        if (canvas) {
          const updateMousePosition = (e: Event) => {
            const mouseEvent = e as MouseEvent;
            const rect = canvas.getBoundingClientRect();

            // Calculate mouse speed for dynamic effects
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const currentTime = Date.now();
            mouseSpeed = Math.sqrt(
              Math.pow(mouseEvent.clientX - lastMouseX, 2) +
              Math.pow(mouseEvent.clientY - lastMouseY, 2)
            );

            mouseX = mouseEvent.clientX - rect.left;
            mouseY = mouseEvent.clientY - rect.top;
            lastMouseX = mouseEvent.clientX;
            lastMouseY = mouseEvent.clientY;

            // Enhanced particle interaction system
            const particles = (window as any).pJSDom[0]?.pJS?.particles?.array;
            if (particles) {
              particles.forEach((particle: any) => {
                if (particle && particle.position) {
                  const dx = mouseX - particle.position.x;
                  const dy = mouseY - particle.position.y;
                  const distance = Math.sqrt(dx * dx + dy * dy);

                  // Multiple interaction zones with different effects
                  if (distance < 50) {
                    // Close proximity - strong attraction
                    particle.vx += (dx / distance) * 0.02;
                    particle.vy += (dy / distance) * 0.02;
                    particle.size.value = Math.min(particle.size.value + 0.1, 6);
                  } else if (distance < 100) {
                    // Medium proximity - gentle influence
                    particle.vx += (dx / distance) * 0.01;
                    particle.vy += (dy / distance) * 0.01;
                  } else if (distance < 150) {
                    // Far proximity - subtle movement
                    particle.vx += (dx / distance) * 0.005;
                    particle.vy += (dy / distance) * 0.005;
                  }

                  // Add turbulence based on mouse speed
                  if (mouseSpeed > 5) {
                    particle.vx += (Math.random() - 0.5) * 0.02;
                    particle.vy += (Math.random() - 0.5) * 0.02;
                  }
                }
              });
            }
          };

          // Enhanced click effects
          const handleClick = (e: Event) => {
            clickCount++;
            const mouseEvent = e as MouseEvent;
            const rect = canvas.getBoundingClientRect();
            const clickX = mouseEvent.clientX - rect.left;
            const clickY = mouseEvent.clientY - rect.top;

            // Create click ripple effect
            const particles = (window as any).pJSDom[0]?.pJS?.particles?.array;
            if (particles) {
              particles.forEach((particle: any) => {
                if (particle && particle.position) {
                  const dx = clickX - particle.position.x;
                  const dy = clickY - particle.position.y;
                  const distance = Math.sqrt(dx * dx + dy * dy);

                  if (distance < 80) {
                    // Push particles away from click
                    particle.vx += (dx / distance) * 0.05;
                    particle.vy += (dy / distance) * 0.05;
                    particle.size.value = Math.max(particle.size.value - 0.5, 1);
                  }
                }
              });
            }
          };

          // Add event listeners for both desktop and mobile
          canvas.addEventListener('mousemove', updateMousePosition);

          // Mobile touch event handling
          const handleTouchMove = (e: Event) => {
            e.preventDefault();
            const touchEvent = e as TouchEvent;
            const touch = touchEvent.touches[0];
            const rect = canvas.getBoundingClientRect();
            mouseX = touch.clientX - rect.left;
            mouseY = touch.clientY - rect.top;

            // Simulate mouse speed for mobile
            mouseSpeed = 3;

            updateMousePosition(e);
          };

          const handleTouchEnd = (e: Event) => {
            e.preventDefault();
            handleClick(e);
          };

          canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
          canvas.addEventListener('touchend', handleTouchEnd, { passive: false });
          canvas.addEventListener('click', handleClick);

          // Add keyboard interactions for extra fun (desktop only)
          const handleKeyPress = (e: KeyboardEvent) => {
            if (e.key === ' ') {
              // Spacebar creates burst effect
              const particles = (window as any).pJSDom[0]?.pJS?.particles?.array;
              if (particles) {
                particles.forEach((particle: any) => {
                  particle.vx += (Math.random() - 0.5) * 0.1;
                  particle.vy += (Math.random() - 0.5) * 0.1;
                });
              }
            }
          };

          // Only add keyboard events on desktop
          if (!isMobile) {
            document.addEventListener('keypress', handleKeyPress);
          }

          // Cleanup function
          return () => {
            canvas.removeEventListener('mousemove', updateMousePosition);
            canvas.removeEventListener('touchmove', handleTouchMove);
            canvas.removeEventListener('click', handleClick);
            canvas.removeEventListener('touchend', handleTouchEnd);
            if (!isMobile) {
              document.removeEventListener('keypress', handleKeyPress);
            }
          };
        }
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

  const handleNewAssessment = () => {
    setAssessmentData(null);
    setCurrentSection('assessment');
    const el = document.getElementById('assessment');
    if (el) el.scrollIntoView({ behavior: 'smooth' });
  };

  const scrollToSection = (sectionId: string) => {
    setCurrentSection(sectionId);
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="App bg-gray-900 text-white min-h-screen overflow-x-hidden">
      {/* Animated Background */}
      <div id="particles-js" className="fixed inset-0 z-0 pointer-events-none"></div>
      
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
          <Results data={assessmentData} onNewAssessment={handleNewAssessment} />
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
