import React from 'react';

const Insights: React.FC = () => {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="text-center mb-12">
        <h2 className="text-4xl font-bold mb-4">AI Model Insights</h2>
        <p className="text-xl text-gray-300">Understanding how our 9 advanced AI models work together</p>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
        <div className="glass-effect rounded-2xl p-6 text-center hover:transform hover:scale-105 transition-all duration-300">
          <div className="text-4xl text-blue-400 mb-4">
            <i className="fas fa-brain"></i>
          </div>
          <h3 className="text-xl font-semibold mb-2">9-Model Ensemble</h3>
          <p className="text-gray-300">Advanced ensemble system combining 9 different AI models for maximum accuracy.</p>
        </div>
        <div className="glass-effect rounded-2xl p-6 text-center hover:transform hover:scale-105 transition-all duration-300">
          <div className="text-4xl text-purple-400 mb-4">
            <i className="fas fa-chart-line"></i>
          </div>
          <h3 className="text-xl font-semibold mb-2">Advanced Features</h3>
          <p className="text-gray-300">32+ engineered features including interactions, risk scores, and health indicators.</p>
        </div>
        <div className="glass-effect rounded-2xl p-6 text-center hover:transform hover:scale-105 transition-all duration-300">
          <div className="text-4xl text-pink-400 mb-4">
            <i className="fas fa-shield-alt"></i>
          </div>
          <h3 className="text-xl font-semibold mb-2">Hyperparameter Tuning</h3>
          <p className="text-gray-300">Advanced hyperparameter optimization with 50+ trials per model for maximum performance.</p>
        </div>
        <div className="glass-effect rounded-2xl p-6 text-center hover:transform hover:scale-105 transition-all duration-300">
          <div className="text-4xl text-green-400 mb-4">
            <i className="fas fa-eye"></i>
          </div>
          <h3 className="text-xl font-semibold mb-2">Feature Selection</h3>
          <p className="text-gray-300">Advanced feature selection using multiple techniques.</p>
        </div>
      </div>
    </div>
  );
};

export default Insights;
