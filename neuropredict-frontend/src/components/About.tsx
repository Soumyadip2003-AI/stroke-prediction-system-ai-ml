import React from 'react';

const About: React.FC = () => {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="grid lg:grid-cols-2 gap-12 items-center">
        <div>
          <h2 className="text-4xl font-bold mb-6">About NeuroPredict</h2>
          <p className="text-xl text-gray-300 mb-8">
            NeuroPredict is an advanced AI-powered stroke risk assessment system featuring
            the Ultimate XGBoost Model with state-of-the-art performance.
            Our system achieves 95.11% accuracy through advanced feature engineering and hyperparameter optimization.
          </p>
          <div className="grid grid-cols-2 gap-4">
            <div className="flex items-center">
              <i className="fas fa-check-circle text-green-400 mr-3"></i>
              <span>95.11% Accuracy Rate</span>
            </div>
            <div className="flex items-center">
              <i className="fas fa-check-circle text-green-400 mr-3"></i>
              <span>Ultimate XGBoost Model</span>
            </div>
            <div className="flex items-center">
              <i className="fas fa-check-circle text-green-400 mr-3"></i>
              <span>Real-time Predictions</span>
            </div>
            <div className="flex items-center">
              <i className="fas fa-check-circle text-green-400 mr-3"></i>
              <span>30+ Engineered Features</span>
            </div>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-6">
          <div className="glass-effect rounded-2xl p-6 text-center">
            <i className="fab fa-python text-4xl text-blue-400 mb-4"></i>
            <div className="font-semibold">Python</div>
          </div>
          <div className="glass-effect rounded-2xl p-6 text-center">
            <i className="fas fa-tree text-4xl text-purple-400 mb-4"></i>
            <div className="font-semibold">XGBoost</div>
          </div>
          <div className="glass-effect rounded-2xl p-6 text-center">
            <i className="fas fa-search text-4xl text-pink-400 mb-4"></i>
            <div className="font-semibold">Optuna</div>
          </div>
          <div className="glass-effect rounded-2xl p-6 text-center">
            <i className="fas fa-chart-line text-4xl text-green-400 mb-4"></i>
            <div className="font-semibold">Feature Engineering</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;
