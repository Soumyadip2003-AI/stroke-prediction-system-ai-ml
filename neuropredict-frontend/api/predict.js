// Serverless function for Vercel deployment
// This is a simplified prediction function for demo purposes

// Simple prediction logic for demo purposes
// In production, this would call your Flask backend
function calculateStrokeRisk(data) {
  let riskScore = 0;

  // Age factor (older = higher risk)
  if (data.age >= 60) riskScore += 30;
  else if (data.age >= 45) riskScore += 20;
  else if (data.age >= 30) riskScore += 10;

  // Gender factor
  if (data.gender === 'Male') riskScore += 5;

  // Medical conditions
  if (data.hypertension === 'Yes') riskScore += 25;
  if (data.heart_disease === 'Yes') riskScore += 25;

  // Glucose level
  if (data.avg_glucose_level >= 200) riskScore += 20;
  else if (data.avg_glucose_level >= 140) riskScore += 10;

  // BMI factor
  if (data.bmi >= 35) riskScore += 20;
  else if (data.bmi >= 30) riskScore += 15;
  else if (data.bmi >= 25) riskScore += 10;

  // Smoking factor
  if (data.smoking_status === 'smokes') riskScore += 15;
  else if (data.smoking_status === 'formerly smoked') riskScore += 10;

  // Marital status and work type (minor factors)
  if (data.ever_married === 'Yes') riskScore += 5;

  // Clamp to 0-100 range
  riskScore = Math.max(0, Math.min(100, riskScore));

  return {
    risk_percentage: Math.round(riskScore * 100) / 100,
    risk_category: getRiskCategory(riskScore),
    confidence: 'High',
    risk_color: getRiskColor(riskScore),
    health_analysis: generateHealthAnalysis(data),
    recommendations: generateRecommendations(riskScore, data)
  };
}

function getRiskCategory(riskScore) {
  if (riskScore >= 70) return 'High Risk';
  if (riskScore >= 40) return 'Moderate Risk';
  return 'Low Risk';
}

function getRiskColor(riskScore) {
  if (riskScore >= 70) return '#EF4444'; // Red
  if (riskScore >= 40) return '#F59E0B'; // Orange
  return '#10B981'; // Green
}

function generateHealthAnalysis(data) {
  const analysis = [];

  if (data.age >= 60) {
    analysis.push('Age is a significant risk factor for stroke');
  }

  if (data.hypertension === 'Yes') {
    analysis.push('Hypertension significantly increases stroke risk');
  }

  if (data.heart_disease === 'Yes') {
    analysis.push('Heart disease is a major stroke risk factor');
  }

  if (data.avg_glucose_level >= 140) {
    analysis.push('Elevated glucose levels indicate diabetes risk');
  }

  if (data.bmi >= 30) {
    analysis.push('High BMI increases cardiovascular risk');
  }

  return analysis;
}

function generateRecommendations(riskScore, data) {
  const recommendations = [];

  if (riskScore >= 40) {
    recommendations.push('Regular blood pressure monitoring recommended');
    recommendations.push('Maintain a healthy diet and exercise routine');
  }

  if (data.smoking_status !== 'never smoked') {
    recommendations.push('Consider smoking cessation programs');
  }

  if (data.bmi >= 25) {
    recommendations.push('Weight management consultation advised');
  }

  recommendations.push('Annual health checkups recommended');
  recommendations.push('Stay hydrated and maintain active lifestyle');

  return recommendations;
}

// Export for Vercel serverless functions
module.exports = async (req, res) => {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const data = req.body;

    // Validate required fields
    const requiredFields = ['age', 'gender', 'ever_married', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'work_type', 'residence_type', 'smoking_status'];

    for (const field of requiredFields) {
      if (!data[field]) {
        return res.status(400).json({
          error: `Missing required field: ${field}`
        });
      }
    }

    // Calculate risk using simple algorithm
    const result = calculateStrokeRisk(data);

    // Return the prediction result
    res.status(200).json(result);

  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({
      error: 'Internal server error',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};
