import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the hero headline and the primary call to action', () => {
  render(<App />);
  expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(/Stroke Risk Prediction/i);
  expect(screen.getByRole('link', { name: /start assessment/i })).toBeInTheDocument();
});
