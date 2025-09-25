import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBrain, faRobot, faMagicWandSparkles } from '@fortawesome/free-solid-svg-icons';

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
              Advanced AI
            </span>
            <br />
            <span className="text-white">Stroke Risk Prediction</span>
          </h1>
          <p className="text-xl text-gray-300 leading-relaxed">
            Experience advanced healthcare AI using supervised and unsupervised learning with 6 models.
            Get personalized stroke risk predictions with 88.56% accuracy and intelligent recommendations.
          </p>

          {/* AI Indicators */}
          <div className="flex flex-wrap items-center justify-start gap-4 mb-4">
            <div className="flex items-center bg-blue-900 bg-opacity-50 px-4 py-2 rounded-full">
              <FontAwesomeIcon icon={faBrain} className="text-blue-400 mr-2" />
              <span className="text-blue-400 text-sm font-medium">Advanced ML Models</span>
            </div>
            <div className="flex items-center bg-green-900 bg-opacity-50 px-4 py-2 rounded-full">
              <span className="text-green-400 text-sm font-medium">High Accuracy</span>
            </div>
            <div className="flex items-center bg-purple-900 bg-opacity-50 px-4 py-2 rounded-full">
              <FontAwesomeIcon icon={faMagicWandSparkles} className="text-purple-400 mr-2" />
              <span className="text-purple-400 text-sm font-medium">Intelligent Analysis</span>
            </div>
          </div>
          <div className="grid grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-400">88.56%</div>
              <div className="text-sm text-gray-400">Accuracy</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-400">6</div>
              <div className="text-sm text-gray-400">AI Models</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-green-400">∞</div>
              <div className="text-sm text-gray-400">Learning</div>
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
            <FontAwesomeIcon icon={faRobot} className="mr-2" />
            <span>Start AI Stroke Assessment</span>
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
