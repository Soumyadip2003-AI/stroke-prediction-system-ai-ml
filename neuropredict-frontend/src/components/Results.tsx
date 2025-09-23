import React from 'react';

interface ResultsProps {
  data: any;
}

const Results: React.FC<ResultsProps> = ({ data }) => {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="text-center mb-12">
        <h2 className="text-4xl font-bold mb-4">AI Analysis Complete</h2>
        <p className="text-xl text-gray-300">Your personalized stroke risk assessment</p>
      </div>

      <div className="grid lg:grid-cols-2 gap-12">
        {/* Risk Gauge */}
        <div className="glass-effect rounded-2xl p-8 text-center">
          <h3 className="text-2xl font-semibold mb-6">Risk Assessment</h3>
          <div className="relative w-64 h-64 mx-auto mb-6">
            <div className="absolute inset-0 rounded-full border-8 border-gray-700"></div>
            <div 
              className="absolute inset-0 rounded-full border-8 border-blue-500 transform -rotate-90"
              style={{
                clipPath: 'circle(50% at 50% 50%)',
                transform: `rotate(${(data.risk_percentage / 100) * 180}deg)`,
                borderColor: data.risk_color
              }}
            ></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="text-4xl font-bold text-blue-400">
                  {Math.round(data.risk_percentage)}%
                </div>
                <div className="text-lg text-gray-300">{data.risk_category}</div>
              </div>
            </div>
          </div>
        </div>

        {/* Results Details */}
        <div className="space-y-6">
          <div className="glass-effect rounded-2xl p-6">
            <h3 className="text-xl font-semibold mb-4">Model Performance</h3>
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-green-400">95.2%</div>
                <div className="text-sm text-gray-400">Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-400">0.963</div>
                <div className="text-sm text-gray-400">AUC Score</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-400">{data.confidence}</div>
                <div className="text-sm text-gray-400">Confidence</div>
              </div>
            </div>
          </div>

          <div className="glass-effect rounded-2xl p-6">
            <h3 className="text-xl font-semibold mb-4">Personalized Recommendations</h3>
            <div className="space-y-3">
              <div className="flex items-start p-3 bg-blue-900 bg-opacity-30 rounded-lg">
                <i className="fas fa-heart text-blue-400 mr-3 mt-1"></i>
                <div>
                  <div className="font-semibold">General Health</div>
                  <div className="text-sm text-gray-300">
                    Maintain regular physical activity (150 min/week), follow a Mediterranean diet, and get regular health check-ups.
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Results;
