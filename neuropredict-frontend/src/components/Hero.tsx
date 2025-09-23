import React from 'react';

interface HeroProps {
  onStartAssessment: () => void;
}

const Hero: React.FC<HeroProps> = ({ onStartAssessment }) => {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
      <div className="grid lg:grid-cols-2 gap-12 items-center">
        <div className="space-y-8">
          <h1 className="text-5xl lg:text-7xl font-bold leading-tight">
            <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              Advanced AI-Powered
            </span>
            <br />
            <span className="text-white">Stroke Risk Prediction</span>
          </h1>
          <p className="text-xl text-gray-300 leading-relaxed">
            Harness the power of cutting-edge machine learning to predict stroke risk with 95%+ accuracy. 
            Get personalized insights and recommendations for better health outcomes.
          </p>
          <div className="grid grid-cols-3 gap-6">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-400">95%+</div>
              <div className="text-sm text-gray-400">Accuracy</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-400">7</div>
              <div className="text-sm text-gray-400">AI Models</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-pink-400">20+</div>
              <div className="text-sm text-gray-400">Features</div>
            </div>
          </div>
          <button
            onClick={onStartAssessment}
            className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white px-8 py-4 rounded-full font-semibold transition-all duration-300 transform hover:scale-105 hover:shadow-lg"
          >
            <span>Start Assessment</span>
            <i className="fas fa-arrow-right ml-2"></i>
          </button>
        </div>
        <div className="flex justify-center">
          <div className="neural-network">
            <div className="neuron"></div>
            <div className="neuron"></div>
            <div className="neuron"></div>
            <div className="neuron"></div>
            <div className="neuron"></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hero;
