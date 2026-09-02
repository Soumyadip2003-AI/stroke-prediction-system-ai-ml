import React from 'react';
import { render, screen } from '@testing-library/react';
import Results from './components/Results';
import Assessment from './components/Assessment';
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

test('results are reachable without relying on colour', () => {
  const { container } = render(<Results data={fixture} />);
  // the gauge must not be colour-only: it needs an accessible label
  const gauge = container.querySelector('svg[role="img"]');
  expect(gauge).toBeTruthy();
  expect(gauge!.getAttribute('aria-label')).toMatch(/percent/i);
  // heading order: no level skipped
  const levels = Array.from(container.querySelectorAll('h1,h2,h3,h4')).map((h) =>
    Number(h.tagName[1])
  );
  levels.forEach((lvl, i) => {
    if (i > 0) expect(lvl - levels[i - 1]).toBeLessThanOrEqual(1);
  });
  // explanations are a real list, not divs
  expect(container.querySelectorAll('ul li').length).toBeGreaterThan(0);
  // buttons are buttons
  expect(screen.getAllByRole('button').length).toBeGreaterThan(0);
});

test('every form control has an accessible name', () => {
  const { container } = render(<Assessment onComplete={() => {}} onLoadingChange={() => {}} />);
  const controls = Array.from(container.querySelectorAll('input,select')) as HTMLElement[];
  const unnamed = controls.filter((el) => {
    if (el.getAttribute('type') === 'radio') return false; // wrapped in <label>
    const id = el.getAttribute('id');
    return !id || !container.querySelector(`label[for="${id}"]`);
  });
  expect(unnamed.map((e) => e.getAttribute('name') || e.tagName)).toEqual([]);
});
