import React from 'react';

const STACK = ['Python', 'XGBoost', 'Optuna', 'Feature engineering', 'React'];

const About: React.FC = () => (
  <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
    <div data-reveal>
      <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">About NeuroPredict</h2>
      <p className="mt-6 text-lg text-fog-300 leading-relaxed max-w-[65ch]">
        NeuroPredict is an AI-powered stroke risk assessment built on a gradient-boosted model
        trained on a public stroke dataset of 5,110 records. It exists to make the factors behind stroke risk
        legible, not to replace a clinician.
      </p>
      <p className="mt-5 text-fog-400 leading-relaxed max-w-[65ch]">
        Your answers are sent to the prediction service, scored, and returned. Nothing is stored,
        and no account is required.
      </p>
    </div>

    {/* Placeholder photography. Swap for a real asset before launch. */}
    <figure data-reveal className="mt-12">
      <img
        src="https://picsum.photos/seed/neuropredict-clinic-consult/1400/700"
        alt="A clinician reviewing health records with a patient"
        className="w-full rounded-card border border-white/9 aspect-[2/1] object-cover"
        loading="lazy"
      />
      <figcaption className="mt-3 text-sm text-fog-400">
        A risk score is a prompt for a conversation with a doctor, not a replacement for one.
      </figcaption>
    </figure>

    <div data-reveal className="mt-12 pt-8 border-t border-white/9">
      <h3 className="text-sm font-medium text-fog-400">Built with</h3>
      <ul className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2">
        {STACK.map((item) => (
          <li key={item} className="text-fog-300 font-medium">
            {item}
          </li>
        ))}
      </ul>
    </div>
  </div>
);

export default About;
