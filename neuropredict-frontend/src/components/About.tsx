import React from 'react';

const About: React.FC = () => {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="grid lg:grid-cols-2 gap-12 items-center">
        <div>
          <h2 className="text-4xl font-bold mb-6">About NeuroPredict</h2>
          <p className="text-xl text-gray-300 mb-8">
            NeuroPredict is an advanced AI-powered stroke risk assessment system that combines 
            cutting-edge machine learning algorithms with comprehensive health data analysis. 
            Our system achieves 88.56% accuracy through ensemble learning and advanced feature engineering.
          </p>
          <div className="grid grid-cols-2 gap-4">
            <div className="flex items-center">
              <i className="fas fa-check-circle text-green-400 mr-3"></i>
              <span>88.56% Accuracy Rate</span>
            </div>
            <div className="flex items-center">
              <i className="fas fa-check-circle text-green-400 mr-3"></i>
              <span>6 Advanced AI Models</span>
            </div>
            <div className="flex items-center">
              <i className="fas fa-check-circle text-green-400 mr-3"></i>
              <span>Real-time Predictions</span>
            </div>
            <div className="flex items-center">
              <i className="fas fa-check-circle text-green-400 mr-3"></i>
              <span>Personalized Recommendations</span>
            </div>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-6">
          <div className="glass-effect rounded-2xl p-6 text-center">
            <i className="fab fa-python text-4xl text-blue-400 mb-4"></i>
            <div className="font-semibold">Python</div>
          </div>
          <div className="glass-effect rounded-2xl p-6 text-center">
            <i className="fas fa-brain text-4xl text-purple-400 mb-4"></i>
            <div className="font-semibold">XGBoost</div>
          </div>
          <div className="glass-effect rounded-2xl p-6 text-center">
            <i className="fas fa-network-wired text-4xl text-pink-400 mb-4"></i>
            <div className="font-semibold">Neural Networks</div>
          </div>
          <div className="glass-effect rounded-2xl p-6 text-center">
            <i className="fas fa-chart-bar text-4xl text-green-400 mb-4"></i>
            <div className="font-semibold">SHAP</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;
