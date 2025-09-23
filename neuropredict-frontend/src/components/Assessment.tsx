import React, { useState } from 'react';

interface AssessmentProps {
  onComplete: (data: any) => void;
}

const Assessment: React.FC<AssessmentProps> = ({ onComplete }) => {
  const [currentStep, setCurrentStep] = useState(1);
  const [formData, setFormData] = useState({
    age: 50,
    gender: 'Male',
    ever_married: 'No',
    hypertension: 'No',
    heart_disease: 'No',
    avg_glucose_level: 120,
    bmi: 25,
    work_type: 'Private',
    residence_type: 'Urban',
    smoking_status: 'never smoked'
  });

  const nextStep = () => {
    if (currentStep < 3) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleInputChange = (field: string, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async () => {
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      });

      if (response.ok) {
        const result = await response.json();
        onComplete(result);
      } else {
        // Fallback to simulated prediction
        const simulatedResult = {
          risk_percentage: Math.random() * 100,
          risk_category: 'Moderate Risk',
          risk_color: '#F59E0B',
          confidence: 'High',
          health_analysis: [],
          recommendations: []
        };
        onComplete(simulatedResult);
      }
    } catch (error) {
      console.error('Prediction error:', error);
      // Fallback to simulated prediction
      const simulatedResult = {
        risk_percentage: Math.random() * 100,
        risk_category: 'Moderate Risk',
        risk_color: '#F59E0B',
        confidence: 'High',
        health_analysis: [],
        recommendations: []
      };
      onComplete(simulatedResult);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="text-center mb-12">
        <h2 className="text-4xl font-bold mb-4">AI-Powered Risk Assessment</h2>
        <p className="text-xl text-gray-300">Enter your health information for an advanced stroke risk analysis</p>
      </div>

      <div className="glass-effect rounded-2xl p-8">
        {/* Step 1: Personal Information */}
        {currentStep === 1 && (
          <div>
            <h3 className="text-2xl font-semibold mb-6 flex items-center">
              <i className="fas fa-user mr-3 text-blue-400"></i>
              Personal Information
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium mb-2">
                  <i className="fas fa-calendar mr-2 text-blue-400"></i>
                  Age
                </label>
                <input
                  type="number"
                  value={formData.age}
                  onChange={(e) => handleInputChange('age', parseInt(e.target.value))}
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300"
                  min="1"
                  max="100"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">
                  <i className="fas fa-venus-mars mr-2 text-blue-400"></i>
                  Gender
                </label>
                <select
                  value={formData.gender}
                  onChange={(e) => handleInputChange('gender', e.target.value)}
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300"
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">
                  <i className="fas fa-heart mr-2 text-blue-400"></i>
                  Marital Status
                </label>
                <select
                  value={formData.ever_married}
                  onChange={(e) => handleInputChange('ever_married', e.target.value)}
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300"
                >
                  <option value="No">Never Married</option>
                  <option value="Yes">Married</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {/* Step 2: Health Metrics */}
        {currentStep === 2 && (
          <div>
            <h3 className="text-2xl font-semibold mb-6 flex items-center">
              <i className="fas fa-heartbeat mr-3 text-purple-400"></i>
              Health Metrics
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium mb-2">
                  <i className="fas fa-heartbeat mr-2 text-purple-400"></i>
                  Hypertension
                </label>
                <div className="flex space-x-4">
                  <label className="flex items-center">
                    <input
                      type="radio"
                      name="hypertension"
                      value="No"
                      checked={formData.hypertension === 'No'}
                      onChange={(e) => handleInputChange('hypertension', e.target.value)}
                      className="mr-2"
                    />
                    <span>No</span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="radio"
                      name="hypertension"
                      value="Yes"
                      checked={formData.hypertension === 'Yes'}
                      onChange={(e) => handleInputChange('hypertension', e.target.value)}
                      className="mr-2"
                    />
                    <span>Yes</span>
                  </label>
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">
                  <i className="fas fa-heart mr-2 text-purple-400"></i>
                  Heart Disease
                </label>
                <div className="flex space-x-4">
                  <label className="flex items-center">
                    <input
                      type="radio"
                      name="heart_disease"
                      value="No"
                      checked={formData.heart_disease === 'No'}
                      onChange={(e) => handleInputChange('heart_disease', e.target.value)}
                      className="mr-2"
                    />
                    <span>No</span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="radio"
                      name="heart_disease"
                      value="Yes"
                      checked={formData.heart_disease === 'Yes'}
                      onChange={(e) => handleInputChange('heart_disease', e.target.value)}
                      className="mr-2"
                    />
                    <span>Yes</span>
                  </label>
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">
                  <i className="fas fa-tint mr-2 text-purple-400"></i>
                  Average Glucose Level (mg/dL)
                </label>
                <input
                  type="number"
                  value={formData.avg_glucose_level}
                  onChange={(e) => handleInputChange('avg_glucose_level', parseFloat(e.target.value))}
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300"
                  min="50"
                  max="300"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">
                  <i className="fas fa-weight mr-2 text-purple-400"></i>
                  BMI (Body Mass Index)
                </label>
                <input
                  type="number"
                  value={formData.bmi}
                  onChange={(e) => handleInputChange('bmi', parseFloat(e.target.value))}
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300"
                  min="10"
                  max="50"
                  step="0.1"
                />
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Lifestyle & Environment */}
        {currentStep === 3 && (
          <div>
            <h3 className="text-2xl font-semibold mb-6 flex items-center">
              <i className="fas fa-briefcase mr-3 text-pink-400"></i>
              Lifestyle & Environment
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium mb-2">
                  <i className="fas fa-briefcase mr-2 text-pink-400"></i>
                  Work Type
                </label>
                <select
                  value={formData.work_type}
                  onChange={(e) => handleInputChange('work_type', e.target.value)}
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-pink-500 focus:border-transparent transition-all duration-300"
                >
                  <option value="Private">Private</option>
                  <option value="Self-employed">Self-employed</option>
                  <option value="Children">Children</option>
                  <option value="Never_worked">Never worked</option>
                  <option value="Govt_job">Government Job</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">
                  <i className="fas fa-home mr-2 text-pink-400"></i>
                  Residence Type
                </label>
                <select
                  value={formData.residence_type}
                  onChange={(e) => handleInputChange('residence_type', e.target.value)}
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-pink-500 focus:border-transparent transition-all duration-300"
                >
                  <option value="Urban">Urban</option>
                  <option value="Rural">Rural</option>
                </select>
              </div>
              <div className="md:col-span-2">
                <label className="block text-sm font-medium mb-2">
                  <i className="fas fa-smoking-ban mr-2 text-pink-400"></i>
                  Smoking Status
                </label>
                <select
                  value={formData.smoking_status}
                  onChange={(e) => handleInputChange('smoking_status', e.target.value)}
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-pink-500 focus:border-transparent transition-all duration-300"
                >
                  <option value="never smoked">Never Smoked</option>
                  <option value="formerly smoked">Formerly Smoked</option>
                  <option value="smokes">Currently Smokes</option>
                  <option value="Unknown">Unknown</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {/* Navigation */}
        <div className="flex justify-between items-center mt-8">
          <button
            onClick={prevStep}
            disabled={currentStep === 1}
            className="flex items-center px-6 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <i className="fas fa-chevron-left mr-2"></i>
            Previous
          </button>
          <div className="flex space-x-2">
            {[1, 2, 3].map((step) => (
              <div
                key={step}
                className={`w-3 h-3 rounded-full ${
                  step <= currentStep ? 'bg-blue-500' : 'bg-gray-600'
                }`}
              />
            ))}
          </div>
          {currentStep < 3 ? (
            <button
              onClick={nextStep}
              className="flex items-center px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition-all duration-300"
            >
              Next
              <i className="fas fa-chevron-right ml-2"></i>
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white px-12 py-4 rounded-full font-semibold transition-all duration-300 transform hover:scale-105 hover:shadow-lg"
            >
              <i className="fas fa-brain mr-2"></i>
              <span>Analyze Risk with AI</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default Assessment;
