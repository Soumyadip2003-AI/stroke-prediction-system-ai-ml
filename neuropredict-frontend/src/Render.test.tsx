import React from 'react';
import { render, screen } from '@testing-library/react';
import Results from './components/Results';
// Inlined deliberately. This lived in src/__fixture.json, which the blanket
// *.json rule in .gitignore silently excluded, so a clean clone failed to
// build: react-scripts type-checks test files too.
const fixture: any = {
  "risk_percentage": 55.09460772917365,
  "risk_category": "Very High Risk",
  "risk_color": "#DC2626",
  "risk_multiple": 11.3,
  "population_base_rate": 4.87,
  "flagged": true,
  "health_analysis": [
    {
      "color": "red",
      "description": "BMI indicates obesity, which significantly increases stroke risk.",
      "icon": "fas fa-exclamation-triangle",
      "risk_level": "High Risk",
      "title": "Obesity (BMI: 48.0)",
      "type": "warning"
    },
    {
      "color": "red",
      "description": "Blood glucose levels indicate diabetes.",
      "icon": "fas fa-exclamation-triangle",
      "risk_level": "High Risk",
      "title": "Diabetic Range (290.0 mg/dL)",
      "type": "warning"
    },
    {
      "color": "red",
      "description": "Age is a significant risk factor for stroke.",
      "icon": "fas fa-exclamation-triangle",
      "risk_level": "High Risk",
      "title": "Advanced Age (82.0 years)",
      "type": "warning"
    }
  ],
  "recommendations": [
    {
      "description": "Work with healthcare providers on a comprehensive weight management plan to reduce BMI below 30.",
      "icon": "fas fa-dumbbell",
      "priority": "high",
      "title": "Weight Management"
    },
    {
      "description": "Consult with an endocrinologist about diabetes management and consider a low-carb diet.",
      "icon": "fas fa-stethoscope",
      "priority": "high",
      "title": "Diabetes Management"
    },
    {
      "description": "Continue prescribed medications and monitor blood pressure regularly. Reduce sodium intake.",
      "icon": "fas fa-heartbeat",
      "priority": "high",
      "title": "Blood Pressure Control"
    },
    {
      "description": "Follow your cardiologist's treatment plan and consider cardiac rehabilitation programs.",
      "icon": "fas fa-heart",
      "priority": "high",
      "title": "Cardiac Care"
    },
    {
      "description": "Join a smoking cessation program immediately. Consider nicotine replacement therapy.",
      "icon": "fas fa-smoking-ban",
      "priority": "high",
      "title": "Smoking Cessation"
    },
    {
      "description": "Maintain regular physical activity (150 min/week), follow a Mediterranean diet, and get regular health check-ups.",
      "icon": "fas fa-heart",
      "priority": "medium",
      "title": "General Health"
    }
  ]
};

test('renders a real API response without dropping anything', () => {
  render(<Results data={fixture} />);
  // the number, the multiple, the category
  expect(screen.getByText(/55%/)).toBeInTheDocument();
  expect(screen.getByText(/11\.3x the average/i)).toBeInTheDocument();
  // the explanations the API sends
  expect(screen.getByText(/What is driving your score/i)).toBeInTheDocument();
  expect(screen.getAllByText(/Obesity/i).length).toBeGreaterThan(0);
  expect(screen.getAllByText(/Advanced Age/i).length).toBeGreaterThan(0);
  // the recommendations
  expect(screen.getByText(/What to do next/i)).toBeInTheDocument();
  expect(screen.getAllByText(/Smoking Cessation/i).length).toBeGreaterThan(0);
  // the tail caveat, which must appear above 30%
  expect(screen.getByText(/overstates/i)).toBeInTheDocument();
  // and it must never claim to be a diagnosis
  expect(screen.getByText(/not\s+a diagnosis/i)).toBeInTheDocument();
});
