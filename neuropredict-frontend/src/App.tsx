import React, { useState, useEffect } from 'react';

import Navigation from './components/Navigation';
import Hero from './components/Hero';
import Assessment from './components/Assessment';
import Results from './components/Results';
import ResultsSkeleton from './components/ResultsSkeleton';
import Insights from './components/Insights';
import About from './components/About';
import Footer from './components/Footer';

function App() {
  const [currentSection, setCurrentSection] = useState('home');
  const [assessmentData, setAssessmentData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Scroll reveal. IntersectionObserver only, never a scroll listener.
  // The js-reveal class is what hides elements in the first place, so if this
  // effect never runs the page still renders fully visible.
  useEffect(() => {
    const root = document.documentElement;
    root.classList.add('js-reveal');

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.15, rootMargin: '0px 0px -40px 0px' }
    );

    document.querySelectorAll('[data-reveal]').forEach((el) => observer.observe(el));

    return () => {
      observer.disconnect();
      root.classList.remove('js-reveal');
    };
  }, [assessmentData, isLoading]);

  const handleAssessmentComplete = (data: any) => {
    setAssessmentData(data);
    setCurrentSection('results');
    window.requestAnimationFrame(() => {
      document.getElementById('results')?.scrollIntoView({ block: 'start' });
    });
  };

  const handleNewAssessment = () => {
    setAssessmentData(null);
    setCurrentSection('assessment');
    document.getElementById('assessment')?.scrollIntoView({ block: 'start' });
  };

  return (
    <div className="bg-ink-900 text-fog-100 min-h-[100dvh] overflow-x-hidden">
      <Navigation currentSection={currentSection} onNavigate={setCurrentSection} />

      <main>
        <section id="home" className="min-h-[100dvh] flex items-center pt-24">
          <Hero />
        </section>

        <section id="assessment" className="py-20 sm:py-28 border-t border-white/5">
          <Assessment onComplete={handleAssessmentComplete} onLoadingChange={setIsLoading} />
        </section>

        {(isLoading || assessmentData) && (
          <section id="results" className="py-20 sm:py-28 border-t border-white/5">
            {isLoading ? (
              <ResultsSkeleton />
            ) : (
              <Results data={assessmentData} onNewAssessment={handleNewAssessment} />
            )}
          </section>
        )}

        <section id="insights" className="py-20 sm:py-28 border-t border-white/5">
          <Insights />
        </section>

        <section id="about" className="py-20 sm:py-28 border-t border-white/5">
          <About />
        </section>
      </main>

      <Footer />
    </div>
  );
}

export default App;
