import React, { useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faBrain,
  faHouse,
  faClipboardCheck,
  faChartLine,
  faCircleInfo,
  faBars,
  faXmark,
} from '@fortawesome/free-solid-svg-icons';

interface NavigationProps {
  currentSection: string;
  onNavigate: (section: string) => void;
}

// Labels are frozen. Only the markup changed: real anchors instead of
// scroll-handler buttons, so every section is linkable and keyboard reachable.
const navItems = [
  { id: 'home', label: 'Home', icon: faHouse },
  { id: 'assessment', label: 'Assessment', icon: faClipboardCheck },
  { id: 'insights', label: 'Insights', icon: faChartLine },
  { id: 'about', label: 'About', icon: faCircleInfo },
];

const Navigation: React.FC<NavigationProps> = ({ currentSection, onNavigate }) => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const handleNavClick = (sectionId: string) => {
    onNavigate(sectionId);
    setIsMobileMenuOpen(false);
  };

  return (
    <nav className="fixed top-0 w-full z-50 bg-ink-900/85 backdrop-blur-md border-b border-white/5">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16 lg:h-[72px]">
          <a href="#home" onClick={() => handleNavClick('home')} className="flex items-center gap-2.5">
            <FontAwesomeIcon icon={faBrain} className="text-xl text-accent" />
            <span className="text-lg font-bold bg-gradient-to-r from-accent to-accent-deep bg-clip-text text-transparent">
              NeuroPredict
            </span>
          </a>

          <div className="hidden md:flex items-center gap-1">
            {navItems.map((item) => (
              <a
                key={item.id}
                href={`#${item.id}`}
                onClick={() => handleNavClick(item.id)}
                aria-current={currentSection === item.id ? 'true' : undefined}
                className={`px-3.5 py-2 rounded-full text-sm font-medium transition-colors duration-300 ease-out-expo ${
                  currentSection === item.id
                    ? 'text-fog-100 bg-white/10'
                    : 'text-fog-400 hover:text-fog-100 hover:bg-white/5'
                }`}
              >
                {item.label}
              </a>
            ))}
          </div>

          <button
            onClick={() => setIsMobileMenuOpen((open) => !open)}
            className="md:hidden text-fog-300 hover:text-fog-100 p-2 -mr-2"
            aria-label={isMobileMenuOpen ? 'Close menu' : 'Open menu'}
            aria-expanded={isMobileMenuOpen}
          >
            <FontAwesomeIcon icon={isMobileMenuOpen ? faXmark : faBars} className="text-xl" />
          </button>
        </div>
      </div>

      {isMobileMenuOpen && (
        <div className="md:hidden border-t border-white/5 bg-ink-900/95 backdrop-blur-md">
          <div className="px-4 py-3">
            {navItems.map((item) => (
              <a
                key={item.id}
                href={`#${item.id}`}
                onClick={() => handleNavClick(item.id)}
                aria-current={currentSection === item.id ? 'true' : undefined}
                className={`flex items-center gap-3 px-4 py-3.5 rounded-input min-h-[48px] transition-colors duration-300 ${
                  currentSection === item.id
                    ? 'text-fog-100 bg-white/10'
                    : 'text-fog-400 hover:text-fog-100 hover:bg-white/5'
                }`}
              >
                <FontAwesomeIcon icon={item.icon} className="w-4" />
                <span className="font-medium">{item.label}</span>
              </a>
            ))}
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navigation;
