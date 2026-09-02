import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBrain, faEnvelope, faPhone } from '@fortawesome/free-solid-svg-icons';

const LINKS = [
  { href: '#home', label: 'Home' },
  { href: '#assessment', label: 'Assessment' },
  { href: '#insights', label: 'Insights' },
  { href: '#about', label: 'About' },
];

const Footer: React.FC = () => (
  <footer className="border-t border-white/9 py-14">
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-[1.5fr_1fr_1fr] gap-10">
        <div>
          <div className="flex items-center gap-2.5">
            <FontAwesomeIcon icon={faBrain} className="text-xl text-accent" />
            <span className="text-lg font-bold">NeuroPredict</span>
          </div>
          <p className="mt-4 text-fog-400 leading-relaxed max-w-[38ch]">
            AI-powered stroke risk assessment for better health outcomes.
          </p>
        </div>

        <nav aria-label="Footer">
          <h2 className="text-sm font-semibold text-fog-100">Quick Links</h2>
          <ul className="mt-4 space-y-2.5">
            {LINKS.map((link) => (
              <li key={link.href}>
                <a href={link.href} className="text-fog-400 hover:text-fog-100 transition-colors duration-200">
                  {link.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>

        <div>
          <h2 className="text-sm font-semibold text-fog-100">Contact</h2>
          <ul className="mt-4 space-y-2.5">
            <li>
              <a
                href="mailto:soumyadip.0202@gmail.com"
                className="flex items-center gap-2.5 text-fog-400 hover:text-fog-100 transition-colors duration-200 break-all"
              >
                <FontAwesomeIcon icon={faEnvelope} className="shrink-0" />
                soumyadip.0202@gmail.com
              </a>
            </li>
            <li>
              <a
                href="tel:+917003153300"
                className="flex items-center gap-2.5 text-fog-400 hover:text-fog-100 transition-colors duration-200"
              >
                <FontAwesomeIcon icon={faPhone} className="shrink-0" />
                +91 7003153300
              </a>
            </li>
          </ul>
        </div>
      </div>

      <div className="border-t border-white/9 mt-12 pt-8 text-center">
        <p className="text-fog-400">&copy; 2025 NeuroPredict. All rights reserved.</p>
        <p className="text-sm text-fog-400 mt-2">
          This tool is for educational purposes only. Consult healthcare professionals for medical advice.
        </p>
      </div>
    </div>
  </footer>
);

export default Footer;
