import React, { useEffect, useRef } from 'react';

// Neuron positions in the SVG's 200x200 coordinate space. The CSS places the
// .neuron elements at the same points as percentages, so the two stay in sync.
const NEURONS: Array<[number, number]> = [
  [20, 20],
  [180, 20],
  [100, 100],
  [20, 180],
  [180, 180],
];

const Hero: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const linesRef = useRef<Array<SVGLineElement | null>>([]);

  // Cursor tracking writes straight to the SVG through refs. No React state,
  // so moving the pointer never re-renders the tree.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const finePointer = window.matchMedia('(pointer: fine)');
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (!finePointer.matches || reducedMotion.matches) return;

    let frame = 0;

    const draw = (clientX: number, clientY: number) => {
      const rect = container.getBoundingClientRect();
      const x = ((clientX - rect.left) / rect.width) * 200;
      const y = ((clientY - rect.top) / rect.height) * 200;
      linesRef.current.forEach((line) => {
        if (!line) return;
        line.setAttribute('x1', String(x));
        line.setAttribute('y1', String(y));
      });
    };

    const handlePointerMove = (event: PointerEvent) => {
      if (frame) cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => draw(event.clientX, event.clientY));
    };

    const handleEnter = () => container.classList.add('is-tracking');
    const handleLeave = () => container.classList.remove('is-tracking');

    container.addEventListener('pointermove', handlePointerMove);
    container.addEventListener('pointerenter', handleEnter);
    container.addEventListener('pointerleave', handleLeave);

    return () => {
      if (frame) cancelAnimationFrame(frame);
      container.removeEventListener('pointermove', handlePointerMove);
      container.removeEventListener('pointerenter', handleEnter);
      container.removeEventListener('pointerleave', handleLeave);
    };
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full">
      <div className="grid grid-cols-1 lg:grid-cols-[1.15fr_0.85fr] gap-12 lg:gap-16 items-center">
        <div data-reveal>
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight leading-[1.05]">
            <span className="bg-gradient-to-r from-accent to-accent-deep bg-clip-text text-transparent">
              Advanced AI
            </span>
            <br />
            <span className="text-fog-100">Stroke Risk Prediction</span>
          </h1>

          <p className="mt-6 text-lg sm:text-xl text-fog-400 leading-relaxed max-w-[52ch]">
            Answer ten questions about your health and get a personalised stroke risk assessment
            with clear next steps.
          </p>

          <a
            href="#assessment"
            className="btn-primary mt-9 inline-flex items-center justify-center rounded-full px-8 py-4 text-lg font-semibold whitespace-nowrap"
          >
            Start Assessment
          </a>
        </div>

        <div className="flex justify-center lg:justify-end">
          <div
            ref={containerRef}
            id="neural-network-container"
            className="neural-network-container relative w-64 h-64 sm:w-80 sm:h-80 lg:w-[26rem] lg:h-[26rem]"
          >
            <svg
              className="absolute inset-0 w-full h-full"
              viewBox="0 0 200 200"
              aria-hidden="true"
              focusable="false"
            >
              <defs>
                <linearGradient id="connectionGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#667eea" stopOpacity="0.35" />
                  <stop offset="100%" stopColor="#764ba2" stopOpacity="0.7" />
                </linearGradient>
              </defs>

              {/* Static lattice */}
              {NEURONS.slice(0, 2)
                .concat(NEURONS.slice(3))
                .map(([x, y]) => (
                  <line
                    key={`static-${x}-${y}`}
                    x1={x}
                    y1={y}
                    x2="100"
                    y2="100"
                    stroke="url(#connectionGradient)"
                    strokeWidth="1"
                  />
                ))}
              <line x1="20" y1="20" x2="180" y2="20" stroke="url(#connectionGradient)" strokeWidth="1" />
              <line x1="20" y1="180" x2="180" y2="180" stroke="url(#connectionGradient)" strokeWidth="1" />

              {/* Cursor-follow links, hidden until the pointer enters */}
              {NEURONS.map(([x, y], index) => (
                <line
                  key={`cursor-${x}-${y}`}
                  ref={(el) => {
                    linesRef.current[index] = el;
                  }}
                  className="cursor-link"
                  x1="100"
                  y1="100"
                  x2={x}
                  y2={y}
                  stroke="#667eea"
                  strokeWidth="1.5"
                />
              ))}
            </svg>

            <div className="neural-network">
              <span className="neuron" />
              <span className="neuron" />
              <span className="neuron" />
              <span className="neuron" />
              <span className="neuron" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hero;
