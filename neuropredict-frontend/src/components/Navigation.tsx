import React, { useState } from 'react';

interface NavigationProps {
  currentSection: string;
  onNavigate: (section: string) => void;
}

const Navigation: React.FC<NavigationProps> = ({ currentSection, onNavigate }) => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const navItems = [
    { id: 'home', label: 'Home', icon: 'fas fa-home' },
    { id: 'assessment', label: 'Assessment', icon: 'fas fa-clipboard-check' },
    { id: 'insights', label: 'Insights', icon: 'fas fa-chart-line' },
    { id: 'about', label: 'About', icon: 'fas fa-info-circle' }
  ];

  const handleMobileMenuToggle = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  const handleNavClick = (sectionId: string) => {
    onNavigate(sectionId);
    setIsMobileMenuOpen(false); // Close mobile menu after navigation
  };

  return (
    <nav className="fixed top-0 w-full z-50 bg-gray-900/90 backdrop-blur-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16 sm:h-20">
          <div className="flex items-center space-x-2">
            <i className="fas fa-brain text-xl sm:text-2xl text-blue-400"></i>
            <span className="text-lg sm:text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              NeuroPredict
            </span>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex space-x-6 lg:space-x-8">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => handleNavClick(item.id)}
                className={`px-3 py-2 rounded-md text-sm lg:text-base font-medium transition-all duration-300 ${
                  currentSection === item.id
                    ? 'text-white bg-blue-600/50'
                    : 'text-gray-300 hover:text-white hover:bg-gray-700/30'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button
              onClick={handleMobileMenuToggle}
              className="mobile-menu-button text-gray-300 hover:text-white p-2 rounded-md transition-all duration-300"
              aria-label="Toggle mobile menu"
            >
              <div className="relative w-6 h-6">
                <span className={`absolute block w-6 h-0.5 bg-current transition-all duration-300 ${
                  isMobileMenuOpen ? 'rotate-45 top-3' : 'top-1'
                }`}></span>
                <span className={`absolute block w-6 h-0.5 bg-current transition-all duration-300 ${
                  isMobileMenuOpen ? 'opacity-0' : 'top-3'
                }`}></span>
                <span className={`absolute block w-6 h-0.5 bg-current transition-all duration-300 ${
                  isMobileMenuOpen ? '-rotate-45 top-3' : 'top-5'
                }`}></span>
              </div>
            </button>
          </div>
        </div>

        {/* Mobile Menu Overlay */}
        {isMobileMenuOpen && (
          <>
            {/* Backdrop */}
            <div
              className="fixed inset-0 bg-black/50 backdrop-blur-sm md:hidden"
              onClick={() => setIsMobileMenuOpen(false)}
              style={{ top: '80px' }}
            ></div>

            {/* Mobile Menu */}
            <div className="absolute top-full left-0 right-0 bg-gray-900/95 backdrop-blur-md border-t border-gray-700 md:hidden">
              <div className="px-4 py-4 space-y-1">
                {navItems.map((item, index) => (
                  <button
                    key={item.id}
                    onClick={() => handleNavClick(item.id)}
                    className={`mobile-nav-item w-full text-left px-4 py-4 rounded-lg transition-all duration-300 ${
                      currentSection === item.id
                        ? 'text-white bg-blue-600/50'
                        : 'text-gray-300 hover:text-white hover:bg-gray-700/30'
                    }`}
                    style={{
                      animationDelay: `${index * 50}ms`,
                      touchAction: 'manipulation',
                      minHeight: '48px'
                    }}
                  >
                    <div className="flex items-center space-x-3">
                      <i className={`${item.icon} text-lg`}></i>
                      <span className="font-medium">{item.label}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </nav>
  );
};

export default Navigation;
