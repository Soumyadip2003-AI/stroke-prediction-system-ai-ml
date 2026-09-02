import React from 'react';

// Skeleton mirrors the shape of Results: gauge, heading, buttons, list.
const ResultsSkeleton: React.FC = () => (
  <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 animate-pulse" aria-busy="true" aria-live="polite">
    <span className="sr-only">Analysing your answers</span>

    <div className="flex flex-col items-center">
      <div className="w-[180px] h-[180px] rounded-full border-[12px] border-white/8" />
      <div className="mt-8 h-9 w-72 max-w-full rounded-input bg-white/8" />
      <div className="mt-4 h-5 w-96 max-w-full rounded-input bg-white/5" />
      <div className="mt-8 flex gap-3">
        <div className="h-12 w-48 rounded-full bg-white/8" />
        <div className="h-12 w-36 rounded-full bg-white/5" />
      </div>
    </div>

    <div className="mt-16">
      <div className="h-6 w-40 rounded-input bg-white/8" />
      {[0, 1, 2].map((i) => (
        <div key={i} className="flex items-start gap-4 py-5 border-b border-white/8 last:border-b-0">
          <div className="w-4 h-4 rounded-full bg-white/8 mt-1 shrink-0" />
          <div className="flex-1">
            <div className="h-4 w-44 rounded-input bg-white/8" />
            <div className="mt-2.5 h-4 w-full max-w-lg rounded-input bg-white/5" />
          </div>
        </div>
      ))}
    </div>
  </div>
);

export default ResultsSkeleton;
