import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import Assessment from './components/Assessment';

// Regression: Number('') is 0, so clearing a numeric box snapped it back to
// "0" and the next keystrokes read as "0140" instead of "140".
test('a numeric field can be cleared and retyped', () => {
  render(<Assessment onComplete={() => {}} onLoadingChange={() => {}} />);

  // step 1 holds age
  const age = screen.getByLabelText(/^Age$/i) as HTMLInputElement;
  fireEvent.change(age, { target: { value: '' } });
  expect(age.value).toBe(''); // must stay empty, not become "0"

  fireEvent.change(age, { target: { value: '82' } });
  expect(age.value).toBe('82'); // not "082"
});

test('glucose can be cleared and retyped without a leading zero', () => {
  render(<Assessment onComplete={() => {}} onLoadingChange={() => {}} />);
  fireEvent.click(screen.getByRole('button', { name: /next/i })); // to Health metrics

  const glucose = screen.getByLabelText(/Average glucose level/i) as HTMLInputElement;
  fireEvent.change(glucose, { target: { value: '' } });
  expect(glucose.value).toBe('');

  fireEvent.change(glucose, { target: { value: '140' } });
  expect(glucose.value).toBe('140'); // this used to be "0140"
});
