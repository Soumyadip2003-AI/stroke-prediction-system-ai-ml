import React from 'react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-gray-800 py-12">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid md:grid-cols-4 gap-8">
          <div>
            <div className="flex items-center space-x-2 mb-4">
              <i className="fas fa-brain text-2xl text-blue-400"></i>
              <span className="text-xl font-bold">NeuroPredict</span>
            </div>
            <p className="text-gray-400">
              Advanced AI-powered stroke risk assessment for better health outcomes.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-4">Quick Links</h4>
            <ul className="space-y-2">
              <li><a href="#home" className="text-gray-400 hover:text-white transition-colors">Home</a></li>
              <li><a href="#assessment" className="text-gray-400 hover:text-white transition-colors">Assessment</a></li>
              <li><a href="#insights" className="text-gray-400 hover:text-white transition-colors">Insights</a></li>
              <li><a href="#about" className="text-gray-400 hover:text-white transition-colors">About</a></li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-4">Technology</h4>
            <ul className="space-y-2">
              <li><span className="text-gray-400">Machine Learning</span></li>
              <li><span className="text-gray-400">AI Models</span></li>
              <li><span className="text-gray-400">Data Science</span></li>
              <li><span className="text-gray-400">Healthcare AI</span></li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-4">Contact</h4>
            <div className="space-y-2">
              <p className="text-gray-400"><i className="fas fa-envelope mr-2"></i> info@neuropredict.ai</p>
              <p className="text-gray-400"><i className="fas fa-phone mr-2"></i> +1 (555) 123-4567</p>
            </div>
          </div>
        </div>
        <div className="border-t border-gray-700 mt-8 pt-8 text-center">
          <p className="text-gray-400">&copy; 2024 NeuroPredict. All rights reserved.</p>
          <p className="text-sm text-gray-500 mt-2">
            This tool is for educational purposes only. Consult healthcare professionals for medical advice.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
