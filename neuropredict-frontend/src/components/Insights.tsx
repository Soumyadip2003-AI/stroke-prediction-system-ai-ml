import React from 'react';

const Insights: React.FC = () => {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="text-center mb-12">
        <h2 className="text-4xl font-bold mb-4">AI Model Insights</h2>
        <p className="text-xl text-gray-300">Understanding how our advanced models work</p>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
        <div className="glass-effect rounded-2xl p-6 text-center hover:transform hover:scale-105 transition-all duration-300">
          <div className="text-4xl text-blue-400 mb-4">
            <i className="fas fa-brain"></i>
          </div>
          <h3 className="text-xl font-semibold mb-2">Ensemble Learning</h3>
          <p className="text-gray-300">Combines 6 different AI algorithms for maximum accuracy.</p>
        </div>
        <div className="glass-effect rounded-2xl p-6 text-center hover:transform hover:scale-105 transition-all duration-300">
          <div className="text-4xl text-purple-400 mb-4">
            <i className="fas fa-chart-line"></i>
          </div>
          <h3 className="text-xl font-semibold mb-2">Feature Engineering</h3>
          <p className="text-gray-300">Advanced preprocessing with 20+ engineered features.</p>
        </div>
        <div className="glass-effect rounded-2xl p-6 text-center hover:transform hover:scale-105 transition-all duration-300">
          <div className="text-4xl text-pink-400 mb-4">
            <i className="fas fa-shield-alt"></i>
          </div>
          <h3 className="text-xl font-semibold mb-2">Cross-Validation</h3>
          <p className="text-gray-300">Robust 5-fold validation ensures reliable results.</p>
        </div>
        <div className="glass-effect rounded-2xl p-6 text-center hover:transform hover:scale-105 transition-all duration-300">
          <div className="text-4xl text-green-400 mb-4">
            <i className="fas fa-eye"></i>
          </div>
          <h3 className="text-xl font-semibold mb-2">Explainable AI</h3>
          <p className="text-gray-300">SHAP values provide transparent insights.</p>
        </div>
      </div>
    </div>
  );
};

export default Insights;
