import React from 'react';

interface ResultsProps {
  data: any;
  onNewAssessment?: () => void;
}

const Results: React.FC<ResultsProps> = ({ data, onNewAssessment }) => {

  const handleDownload = () => {
    console.log('Download button clicked');
    try {
      const rows = [
        ['Generated At', new Date().toISOString()],
        ['Risk Percentage', `${Math.round(data?.risk_percentage ?? 0)}%`],
        ['Risk Category', `${data?.risk_category ?? ''}`],
        ['Confidence', `${data?.confidence ?? ''}`],
      ];
      const csvContent = rows.map(r => r.join(',')).join('\n');
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.setAttribute('download', 'stroke_assessment_result.csv');
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      console.log('Download completed successfully');
    } catch (e) {
      console.error('Download failed', e);
    }
  };

  const handleNew = () => {
    console.log('New Assessment button clicked');
    console.log('onNewAssessment function:', onNewAssessment);
    if (onNewAssessment) {
      console.log('Calling onNewAssessment function');
      onNewAssessment();
    } else {
      console.log('No onNewAssessment function, scrolling manually');
      const el = document.getElementById('assessment');
      if (el) {
        console.log('Found assessment element, scrolling');
        el.scrollIntoView({ behavior: 'smooth' });
      } else {
        console.log('Assessment element not found');
      }
      window.setTimeout(() => window.scrollTo({ top: 0, behavior: 'smooth' }), 300);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="text-center mb-12">
        <h2 className="text-4xl font-bold mb-4">AI Stroke Risk Analysis Complete</h2>
        <p className="text-xl text-gray-300">Your personalized stroke risk assessment using 9 Advanced AI Models with 95.11% accuracy</p>

        <div className="mt-6 flex items-center justify-center gap-4 z-10 relative">
          <button type="button" onClick={handleDownload} className="px-5 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold shadow cursor-pointer transition-all duration-200 hover:scale-105">
            Download Results
          </button>
          <button type="button" onClick={handleNew} className="px-5 py-2.5 rounded-lg bg-gray-700 hover:bg-gray-600 text-white font-semibold shadow cursor-pointer transition-all duration-200 hover:scale-105">
            New Assessment
          </button>
        </div>
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
                transform: `rotate(${((data?.risk_percentage ?? 0) / 100) * 180}deg)`,
                borderColor: data?.risk_color ?? '#EF4444'
              }}
            ></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="text-4xl font-bold text-blue-400">
                  {Math.round(data?.risk_percentage ?? 0)}%
                </div>
                <div className="text-lg text-gray-300">{data?.risk_category ?? ''}</div>
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
                <div className="text-2xl font-bold text-green-400">95.11%</div>
                <div className="text-sm text-gray-400">Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-400">0.846</div>
                <div className="text-sm text-gray-400">ROC-AUC</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-400">9 Models</div>
                <div className="text-sm text-gray-400">Ensemble</div>
              </div>
            </div>
          </div>

          <div className="glass-effect rounded-2xl p-6">
            <h3 className="text-xl font-semibold mb-4">Personalized Recommendations</h3>
            <div className="space-y-3">
              {data?.recommendations && data.recommendations.length > 0 ? (
                data.recommendations.map((rec: any, index: number) => (
                  <div key={index} className="flex items-start p-3 bg-blue-900 bg-opacity-30 rounded-lg">
                    <i className={`${rec.icon || 'fas fa-heart'} text-blue-400 mr-3 mt-1`}></i>
                    <div>
                      <div className="font-semibold">{rec.title || 'Recommendation'}</div>
                      <div className="text-sm text-gray-300">
                        {rec.description || 'No description available'}
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="flex items-start p-3 bg-blue-900 bg-opacity-30 rounded-lg">
                  <i className="fas fa-heart text-blue-400 mr-3 mt-1"></i>
                  <div>
                    <div className="font-semibold">General Health</div>
                    <div className="text-sm text-gray-300">
                      Maintain regular physical activity (150 min/week), follow a Mediterranean diet, and get regular health check-ups.
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

        </div>
      </div>

    </div>
  );
};

export default Results;
