import React from 'react';
import { render, screen } from '@testing-library/react';
import Results from './components/Results';
import fixture from './__fixture.json';

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
