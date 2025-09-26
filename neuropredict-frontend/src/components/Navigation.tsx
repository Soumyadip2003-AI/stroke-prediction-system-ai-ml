import React, { useState, useEffect } from 'react';

interface NavigationProps {
  currentSection: string;
  onNavigate: (section: string) => void;
}

const Navigation: React.FC<NavigationProps> = ({ currentSection, onNavigate }) => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const navItems = [
    { id: 'home', label: 'Home' },
    { id: 'assessment', label: 'Assessment' },
    { id: 'insights', label: 'Insights' },
    { id: 'about', label: 'About' }
  ];

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  const closeMobileMenu = () => {
    setIsMobileMenuOpen(false);
  };

  // Close mobile menu on window resize to desktop
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 768) {
        setIsMobileMenuOpen(false);
      }
    };

    // Close mobile menu when clicking outside
    const handleClickOutside = (event: MouseEvent | TouchEvent) => {
      if (isMobileMenuOpen && !(event.target as Element).closest('nav')) {
        setIsMobileMenuOpen(false);
      }
    };

    // Close mobile menu on escape key
    const handleEscapeKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && isMobileMenuOpen) {
        setIsMobileMenuOpen(false);
      }
    };

    // Handle touch events for mobile devices
    const handleTouchStart = (event: TouchEvent) => {
      if (isMobileMenuOpen && !(event.target as Element).closest('nav')) {
        setIsMobileMenuOpen(false);
      }
    };

    // Prevent body scrolling when mobile menu is open
    if (isMobileMenuOpen) {
      document.body.style.overflow = 'hidden';
      document.body.style.position = 'fixed';
      document.body.style.width = '100%';
    } else {
      document.body.style.overflow = '';
      document.body.style.position = '';
      document.body.style.width = '';
    }

    window.addEventListener('resize', handleResize);
    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('touchstart', handleTouchStart, { passive: true });
    document.addEventListener('keydown', handleEscapeKey);

    return () => {
      window.removeEventListener('resize', handleResize);
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('touchstart', handleTouchStart);
      document.removeEventListener('keydown', handleEscapeKey);
      document.body.style.overflow = '';
      document.body.style.position = '';
      document.body.style.width = '';
    };
  }, [isMobileMenuOpen]);

  // Debug: Log navigation state
  console.log('Navigation rendered, isMobileMenuOpen:', isMobileMenuOpen);

  return (
    <nav className="fixed top-0 w-full z-50 bg-gray-900/90 backdrop-blur-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-2">
            <i className="fas fa-brain text-2xl text-blue-400"></i>
            <span className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              NeuroPredict
            </span>
          </div>
          <div className="hidden md:flex space-x-8">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => onNavigate(item.id)}
                className={`transition-colors duration-300 ${
                  currentSection === item.id
                    ? 'text-white'
                    : 'text-gray-300 hover:text-white'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
          <div className="md:hidden">
            <button
              onClick={() => {
                console.log('Mobile menu button clicked! Current state:', isMobileMenuOpen);
                toggleMobileMenu();
              }}
              className="mobile-menu-button p-3 text-gray-200 hover:text-white focus:outline-none focus:text-white transition-all duration-300 rounded-lg hover:bg-gray-700/50 active:bg-gray-600 border border-gray-600"
              aria-label="Toggle mobile menu"
              style={{
                touchAction: 'manipulation',
                minHeight: '48px',
                minWidth: '48px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: 'rgba(55, 65, 81, 0.5)'
              }}
            >
              <span
                className="text-2xl font-bold transition-transform duration-300"
                style={{ pointerEvents: 'none' }}
              >
                {isMobileMenuOpen ? '✕' : '☰'}
              </span>
            </button>
          </div>
        </div>
        {/* Mobile menu */}
        {isMobileMenuOpen && (() => {
          console.log('Mobile menu rendered!');
          console.log('Rendering mobile menu items:', navItems);
          return (
            <div className="md:hidden absolute top-full left-0 w-full bg-gray-900/95 backdrop-blur-md border-t border-gray-600 shadow-xl" style={{ zIndex: 999 }}>
              <div className="px-4 pt-6 pb-8 space-y-3">
                {navItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => {
                    console.log('Mobile menu item clicked:', item.id);
                    onNavigate(item.id);
                    closeMobileMenu();
                  }}
                  className={`block w-full text-left px-6 py-4 text-xl font-medium transition-all duration-300 rounded-lg ${
                    currentSection === item.id
                      ? 'text-white bg-blue-600 shadow-lg'
                      : 'text-gray-200 hover:text-white hover:bg-gray-700/80 active:bg-gray-600'
                  }`}
                  style={{
                    touchAction: 'manipulation',
                    minHeight: '56px',
                    border: '1px solid rgba(255,255,255,0.1)'
                  }}
                >
                  {item.label}
                </button>
              ))}
              </div>
            </div>
          );
        })()}
      </div>
    </nav>
  );
};

export default Navigation;
