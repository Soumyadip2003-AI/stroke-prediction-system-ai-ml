import React from 'react';

const LoadingOverlay: React.FC = () => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="glass-effect rounded-2xl p-8 text-center">
        <div className="text-6xl text-blue-400 mb-4 animate-spin">
          <i className="fas fa-brain"></i>
        </div>
        <div className="text-xl font-semibold mb-2">Analyzing with AI...</div>
        <div className="w-64 bg-gray-700 rounded-full h-2">
          <div className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full animate-pulse"></div>
        </div>
      </div>
    </div>
  );
};

export default LoadingOverlay;
