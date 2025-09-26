import React, { useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBars, faTimes } from '@fortawesome/free-solid-svg-icons';

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
    console.log('Mobile menu toggle clicked, current state:', isMobileMenuOpen);
    setIsMobileMenuOpen(!isMobileMenuOpen);
    console.log('Mobile menu new state:', !isMobileMenuOpen);
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
    const handleClickOutside = (event: MouseEvent) => {
      if (isMobileMenuOpen && !(event.target as Element).closest('nav')) {
        console.log('Closing mobile menu due to click outside');
        setIsMobileMenuOpen(false);
      }
    };

    // Close mobile menu on escape key
    const handleEscapeKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && isMobileMenuOpen) {
        console.log('Closing mobile menu due to escape key');
        setIsMobileMenuOpen(false);
      }
    };

    // Handle touch events for mobile devices
    const handleTouchStart = (event: TouchEvent) => {
      if (isMobileMenuOpen && !(event.target as Element).closest('nav')) {
        console.log('Closing mobile menu due to touch outside');
        setIsMobileMenuOpen(false);
      }
    };

    // Prevent body scrolling when mobile menu is open
    if (isMobileMenuOpen) {
      document.body.style.overflow = 'hidden';
      document.body.classList.add('mobile-menu-open');
    } else {
      document.body.style.overflow = 'unset';
      document.body.classList.remove('mobile-menu-open');
    }

    window.addEventListener('resize', handleResize);
    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleEscapeKey);
    document.addEventListener('touchstart', handleTouchStart, { passive: false });

    return () => {
      window.removeEventListener('resize', handleResize);
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEscapeKey);
      document.removeEventListener('touchstart', handleTouchStart);
      document.body.style.overflow = 'unset';
      document.body.classList.remove('mobile-menu-open');
    };
  }, [isMobileMenuOpen]);

  return (
    <nav className="fixed top-0 w-full z-[100] glass-effect">
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
              onClick={toggleMobileMenu}
              className="mobile-menu-button p-2 text-gray-300 hover:text-white focus:outline-none focus:text-white transition-all duration-300 rounded-md hover:bg-gray-700/30 active:bg-gray-600/50"
              aria-label="Toggle mobile menu"
              style={{
                touchAction: 'manipulation',
                minHeight: '44px',
                minWidth: '44px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              <FontAwesomeIcon
                icon={isMobileMenuOpen ? faTimes : faBars}
                className="text-xl transition-transform duration-300"
                style={{ pointerEvents: 'none' }}
              />
            </button>
          </div>
        </div>
        {/* Mobile menu */}
        {isMobileMenuOpen && (
          <div className="md:hidden absolute top-full left-0 w-full bg-gray-900/98 backdrop-blur-md border-t border-gray-700 shadow-lg">
            <div className="px-4 pt-4 pb-6 space-y-2 max-h-[80vh] overflow-y-auto">
              {navItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => {
                    onNavigate(item.id);
                    closeMobileMenu();
                  }}
                  className={`mobile-nav-item block w-full text-left px-4 py-3 rounded-lg text-lg font-medium transition-all duration-300 ${
                    currentSection === item.id
                      ? 'text-white bg-blue-600/70 shadow-md'
                      : 'text-gray-300 hover:text-white hover:bg-gray-700/60 active:bg-gray-600/60'
                  }`}
                  style={{ touchAction: 'manipulation' }}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navigation;
