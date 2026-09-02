import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCircleCheck, faDownload, faRotateLeft } from '@fortawesome/free-solid-svg-icons';

interface ResultsProps {
  data: any;
  onNewAssessment?: () => void;
}

const RADIUS = 70;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

const Results: React.FC<ResultsProps> = ({ data, onNewAssessment }) => {
  const percentage = Math.min(100, Math.max(0, Math.round(data?.risk_percentage ?? 0)));
  const colour = data?.risk_color ?? '#667eea';

  const handleDownload = () => {
    const rows = [
      ['Generated At', new Date().toISOString()],
      ['Risk Percentage', `${percentage}%`],
      ['Risk Category', `${data?.risk_category ?? ''}`],
      ['Confidence', `${data?.confidence ?? ''}`],
    ];
    const blob = new Blob([rows.map((r) => r.join(',')).join('\n')], {
      type: 'text/csv;charset=utf-8;',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.setAttribute('download', 'stroke_assessment_result.csv');
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const recommendations =
    data?.recommendations?.length > 0
      ? data.recommendations
      : [
          {
            title: 'General health',
            description:
              'Maintain regular physical activity (150 min/week), follow a Mediterranean diet, and get regular health check-ups.',
          },
        ];

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <div data-reveal className="flex flex-col items-center text-center">
        <svg width="180" height="180" viewBox="0 0 180 180" role="img" aria-label={`Estimated stroke risk: ${percentage} percent, ${data?.risk_category ?? 'unclassified'}`}>
          <circle cx="90" cy="90" r={RADIUS} fill="none" stroke="rgba(255,255,255,0.09)" strokeWidth="12" />
          <circle
            cx="90"
            cy="90"
            r={RADIUS}
            fill="none"
            stroke={colour}
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={CIRCUMFERENCE * (1 - percentage / 100)}
            transform="rotate(-90 90 90)"
          />
          <text x="90" y="84" textAnchor="middle" className="fill-fog-100" style={{ fontSize: 34, fontWeight: 700 }}>
            {percentage}%
          </text>
          <text x="90" y="108" textAnchor="middle" className="fill-fog-400" style={{ fontSize: 13 }}>
            {data?.risk_category ?? ''}
          </text>
        </svg>

        <h2 className="mt-8 text-3xl sm:text-4xl font-bold tracking-tight">Your risk assessment</h2>
        <p className="mt-4 text-lg text-fog-400 max-w-[54ch]">
          An estimate based on the ten answers you gave. It is not a diagnosis.
          {data?.confidence ? ` Model confidence: ${data.confidence}.` : ''}
        </p>

        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <button
            type="button"
            onClick={handleDownload}
            className="btn-primary rounded-full px-6 py-3 min-h-[48px] flex items-center gap-2.5 font-semibold whitespace-nowrap"
          >
            <FontAwesomeIcon icon={faDownload} />
            Download Results
          </button>
          <button
            type="button"
            onClick={onNewAssessment}
            className="btn-quiet rounded-full px-6 py-3 min-h-[48px] flex items-center gap-2.5 font-medium whitespace-nowrap"
          >
            <FontAwesomeIcon icon={faRotateLeft} />
            Start over
          </button>
        </div>
      </div>

      <div data-reveal className="mt-16">
        <h3 className="text-xl font-semibold">What to do next</h3>
        <ul className="mt-5">
          {recommendations.map((rec: any, index: number) => (
            <li
              key={index}
              className="flex items-start gap-4 py-5 border-b border-white/8 last:border-b-0"
            >
              <FontAwesomeIcon icon={faCircleCheck} className="text-accent mt-1 shrink-0" />
              <div>
                <p className="font-semibold">{rec.title || 'Recommendation'}</p>
                <p className="mt-1 text-fog-400 leading-relaxed max-w-[62ch]">
                  {rec.description || 'No description available.'}
                </p>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default Results;
