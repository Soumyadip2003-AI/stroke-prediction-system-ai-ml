import React, { useState } from 'react';

interface AssessmentProps {
  onComplete: (data: any) => void;
}

const Assessment: React.FC<AssessmentProps> = ({ onComplete }) => {
  const [currentStep, setCurrentStep] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const API_BASE = (process.env.REACT_APP_API_BASE as string) || 'http://localhost:5002';

  const [formData, setFormData] = useState({
    age: 55,
    gender: 'Male',
    ever_married: 'Yes',
    hypertension: 'Yes',
    heart_disease: 'No',
    avg_glucose_level: 150,
    bmi: 29,
    work_type: 'Private',
    residence_type: 'Urban',
    smoking_status: 'smokes'
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
    console.log('Submit button clicked');
    console.log('Form data:', formData);
    setIsSubmitting(true);
    try {
      console.log('Making API call to:', `${API_BASE}/api/predict`);
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      });

      console.log('Response status:', response.status);
      if (response.ok) {
        const result = await response.json();
        console.log('API response:', result);
        onComplete(result);
      } else {
        console.error('API error, showing error message');
        const errorText = await response.text();
        console.error('Error response:', errorText);
        alert('Prediction service is temporarily unavailable. Please try again later.');
      }
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Network error. Please check your connection and try again.');
    } finally {
      // Always reset the submitting state
      setIsSubmitting(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="text-center mb-12 sm:mb-16">
        <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold mb-6 hover:text-blue-400 transition-colors duration-300 cursor-pointer">
          AI-Powered Stroke Risk Assessment
        </h2>
        <p className="text-lg sm:text-xl lg:text-2xl text-gray-300 leading-relaxed">
          Enter your health information for an advanced stroke risk analysis using{' '}
          <span className="text-blue-400 font-semibold hover:text-purple-400 transition-colors duration-300 cursor-pointer">
            9 Advanced AI Models
          </span>{' '}
          with{' '}
          <span className="text-green-400 font-semibold hover:text-blue-400 transition-colors duration-300 cursor-pointer">
            95%+ accuracy
          </span>
        </p>
      </div>

      <div className="glass-effect rounded-2xl p-6 sm:p-8">
        {/* Step 1: Personal Information */}
        {currentStep === 1 && (
          <div>
            <h3 className="text-xl sm:text-2xl lg:text-3xl font-semibold mb-6 sm:mb-8 flex items-center">
              <i className="fas fa-user mr-3 text-blue-400"></i>
              Personal Information
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              <div>
                <label className="block text-base sm:text-lg font-medium mb-3">
                  <i className="fas fa-calendar mr-2 text-blue-400"></i>
                  Age
                </label>
                <input
                  type="number"
                  value={formData.age}
                  onChange={(e) => handleInputChange('age', parseInt(e.target.value))}
                  className="w-full px-4 py-4 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300 text-lg min-h-[48px]"
                  min="1"
                  max="100"
                />
              </div>
              <div>
                <label className="block text-base sm:text-lg font-medium mb-3">
                  <i className="fas fa-venus-mars mr-2 text-blue-400"></i>
                  Gender
                </label>
                <select
                  value={formData.gender}
                  onChange={(e) => handleInputChange('gender', e.target.value)}
                  className="w-full px-4 py-4 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300 text-lg min-h-[48px]"
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
              <div>
                <label className="block text-base sm:text-lg font-medium mb-3">
                  <i className="fas fa-heart mr-2 text-blue-400"></i>
                  Marital Status
                </label>
                <select
                  value={formData.ever_married}
                  onChange={(e) => handleInputChange('ever_married', e.target.value)}
                  className="w-full px-4 py-4 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300 text-lg min-h-[48px]"
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
        <div className="flex flex-col sm:flex-row justify-between items-center mt-10 sm:mt-12 space-y-4 sm:space-y-0">
          <button
            onClick={prevStep}
            disabled={currentStep === 1 || isSubmitting}
            className="flex items-center justify-center px-6 py-4 bg-gray-700 hover:bg-gray-600 rounded-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed min-w-[120px] min-h-[48px] text-lg"
          >
            <i className="fas fa-chevron-left mr-2"></i>
            Previous
          </button>
          <div className="flex space-x-3">
            {[1, 2, 3].map((step) => (
              <div
                key={step}
                className={`w-4 h-4 rounded-full transition-all duration-300 ${
                  step <= currentStep ? 'bg-blue-500 scale-110' : 'bg-gray-600'
                }`}
              />
            ))}
          </div>
          {currentStep < 3 ? (
            <button
              onClick={nextStep}
              disabled={isSubmitting}
              className="flex items-center justify-center px-8 py-4 bg-blue-600 hover:bg-blue-700 rounded-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed min-w-[120px] min-h-[48px] text-lg"
            >
              Next
              <i className="fas fa-chevron-right ml-2"></i>
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={isSubmitting}
              className={`bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white px-8 sm:px-10 lg:px-12 py-4 sm:py-5 rounded-full font-semibold transition-all duration-300 transform hover:scale-105 hover:shadow-lg hover:shadow-blue-500/30 disabled:opacity-50 disabled:cursor-not-allowed z-10 relative overflow-hidden group min-w-[200px] min-h-[56px] flex items-center justify-center ${
                isSubmitting ? 'animate-pulse' : 'hover:animate-bounce'
              }`}
            >
              {/* Button ripple effect */}
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent transform -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1000 ease-out"></div>

              <div className="relative flex items-center justify-center">
                <i className={`fas fa-brain mr-3 transition-transform duration-300 ${isSubmitting ? 'animate-spin' : 'group-hover:rotate-12'}`}></i>
                <span className="relative text-base sm:text-lg">
                  {isSubmitting ? '🧠 Analyzing with AI...' : '🚀 Get AI Stroke Risk Assessment'}
                </span>
                {!isSubmitting && (
                  <span className="ml-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                    ✨
                  </span>
                )}
              </div>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default Assessment;
