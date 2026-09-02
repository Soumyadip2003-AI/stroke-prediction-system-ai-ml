import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faCircleCheck,
  faDownload,
  faRotateLeft,
  faTriangleExclamation,
} from '@fortawesome/free-solid-svg-icons';

interface ResultsProps {
  data: any;
  onNewAssessment?: () => void;
}

const RADIUS = 70;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

const Results: React.FC<ResultsProps> = ({ data, onNewAssessment }) => {
  // risk_percentage is a calibrated probability: people the model scores at
  // 10-15% had a 12.9% stroke rate in the data. So it tops out near 26%, not
  // 100%. Showing that on a 0-100 arc would leave someone at genuine high risk
  // looking a quarter full, so the arc tracks the multiple of the population
  // average (capped at 5x) while the number stays the true probability.
  const percentage = Math.max(0, data?.risk_percentage ?? 0);
  const baseRate = data?.population_base_rate ?? 4.87;
  // Derive rather than defaulting to 0: a backend that predates risk_multiple
  // would otherwise report "below average" for a high-risk result.
  const multiple =
    data?.risk_multiple ?? (baseRate > 0 ? Math.round((percentage / baseRate) * 10) / 10 : 0);
  const arcFraction = Math.min(1, multiple / 5);
  const colour = data?.risk_color ?? '#667eea';

  const handleDownload = () => {
    // 'Confidence' used to be here. The API always returned "Low" for every
    // user once probabilities were calibrated, because it binned by risk
    // magnitude rather than measuring confidence. Replaced with the multiple
    // of the population rate, which is the number that gives the percentage
    // meaning.
    const rows = [
      ['Generated At', new Date().toISOString()],
      ['Estimated Stroke Risk', `${percentage.toFixed(1)}%`],
      ['Population Average', `${baseRate}%`],
      ['Times Average Risk', `${multiple}x`],
      ['Risk Category', `${data?.risk_category ?? ''}`],
      ['Note', 'Calibrated screening estimate, not a diagnosis'],
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

  const analysis: any[] = Array.isArray(data?.health_analysis) ? data.health_analysis : [];

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
        <svg width="180" height="180" viewBox="0 0 180 180" role="img" aria-label={`Estimated stroke risk: ${percentage.toFixed(1)} percent, ${multiple} times the population average, ${data?.risk_category ?? 'unclassified'}`}>
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
            strokeDashoffset={CIRCUMFERENCE * (1 - arcFraction)}
            transform="rotate(-90 90 90)"
          />
          <text x="90" y="84" textAnchor="middle" className="fill-fog-100" style={{ fontSize: 32, fontWeight: 700 }}>
            {percentage < 1 ? percentage.toFixed(1) : Math.round(percentage)}%
          </text>
          <text x="90" y="108" textAnchor="middle" className="fill-fog-400" style={{ fontSize: 13 }}>
            {data?.risk_category ?? ''}
          </text>
        </svg>

        <h2 className="mt-8 text-3xl sm:text-4xl font-bold tracking-tight">Your risk assessment</h2>
        <p className="mt-4 text-2xl font-semibold text-fog-100">
          {multiple >= 1
            ? `${multiple}x the average person's risk`
            : `Below the average person's risk`}
        </p>
        <p className="mt-3 text-fog-400 max-w-[56ch]">
          The average across the 5,110 people in the dataset is {baseRate}%. This estimate is
          calibrated: of people scored around 15%, about 15% went on to have a stroke.
          {percentage > 30
            ? ' Above 30% the estimate overstates, because few people in the data score that high.'
            : ''}{' '}
          It is not a diagnosis.
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

      {/* The API already computes which factors drove the score. It was being
          sent and dropped, leaving the user a bare percentage with no
          explanation of where it came from. */}
      {analysis.length > 0 && (
        <div data-reveal className="mt-16">
          <h3 className="text-xl font-semibold">What is driving your score</h3>
          <ul className="mt-5 grid gap-3 sm:grid-cols-2">
            {analysis.map((item: any, index: number) => {
              const warning = item?.type === 'warning';
              return (
                <li
                  key={index}
                  className={`flex items-start gap-3 rounded-card border p-4 ${
                    warning ? 'border-[#EF4444]/30 bg-[#EF4444]/8' : 'border-accent/25 bg-accent/8'
                  }`}
                >
                  <FontAwesomeIcon
                    icon={warning ? faTriangleExclamation : faCircleCheck}
                    className={`mt-0.5 shrink-0 ${warning ? 'text-[#f8b4c0]' : 'text-accent'}`}
                  />
                  <div>
                    <p className="font-medium">{item?.title}</p>
                    <p className="mt-1 text-sm text-fog-400 leading-relaxed">{item?.description}</p>
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
      )}

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
