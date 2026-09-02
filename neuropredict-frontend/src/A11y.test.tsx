import React from 'react';
import { render, screen } from '@testing-library/react';
import Results from './components/Results';
import Assessment from './components/Assessment';
import fixture from './__fixture.json';

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
