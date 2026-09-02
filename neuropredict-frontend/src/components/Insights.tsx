import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faScaleUnbalanced, faTriangleExclamation } from '@fortawesome/free-solid-svg-icons';

// Bento: 4 items, 4 cells, no empty cells. Row 1 is 2+1, row 2 is 1+2.
// Every figure below is measured by ml/train_stroke_model.py on a held-out
// split and written to model_metadata.json. Nothing here is asserted by hand.
const Insights: React.FC = () => (
  <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
    <div data-reveal className="mb-10 sm:mb-14">
      <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">How the model works</h2>
      <p className="mt-4 text-lg text-fog-400 max-w-[58ch]">
        A single calibrated model weighs 21 features derived from your ten answers, tuned to
        catch strokes rather than to look accurate.
      </p>
    </div>

    <div data-reveal className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div className="md:col-span-2 relative overflow-hidden rounded-card border border-white/9 min-h-[280px] flex flex-col justify-end p-7">
        {/* Placeholder photography. Swap for a real asset before launch. */}
        <img
          src="https://picsum.photos/seed/neuropredict-model-lab/900/600"
          alt=""
          aria-hidden="true"
          className="absolute inset-0 w-full h-full object-cover opacity-25"
          loading="lazy"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-ink-900 via-ink-900/85 to-ink-900/40" />
        <div className="relative">
          <h3 className="text-2xl font-semibold">Measured performance</h3>
          <dl className="mt-6 flex flex-wrap gap-x-10 gap-y-4">
            <div>
              <dt className="text-xs text-fog-400">ROC-AUC</dt>
              <dd className="text-2xl font-bold text-accent">0.84</dd>
            </div>
            <div>
              <dt className="text-xs text-fog-400">Strokes caught</dt>
              <dd className="text-2xl font-bold text-accent">84%</dd>
            </div>
            <div>
              <dt className="text-xs text-fog-400">Features</dt>
              <dd className="text-2xl font-bold text-accent">21</dd>
            </div>
          </dl>
          <p className="mt-5 text-xs text-fog-400 max-w-[54ch]">
            Held-out 20% split of the public stroke dataset (5,110 records), never seen during
            training or threshold fitting. Across 20 different splits the figure varies
            0.84 plus or minus 0.02, so read it as a range, not a
            precise value. Probabilities are calibrated: below 25%, where 98% of people land, the
            percentage is accurate to within about one point, and above 30% it overstates.
            Research figures, not a clinical validation.
          </p>
        </div>
      </div>

      <div className="surface surface-interactive rounded-card p-7">
        <FontAwesomeIcon icon={faScaleUnbalanced} className="text-accent text-xl" />
        <h3 className="mt-5 text-lg font-semibold">Why not accuracy</h3>
        <p className="mt-2.5 text-fog-400 leading-relaxed">
          Only 4.9% of the dataset had a stroke, so always answering "no" scores 95%. That number
          measures nothing, so it is not used here.
        </p>
      </div>

      <div className="surface surface-interactive rounded-card p-7">
        <h3 className="text-lg font-semibold">What a flag costs</h3>
        <p className="mt-2.5 text-fog-400 leading-relaxed">
          Catching 84% of strokes means flagging people who will not have one. Roughly 1 in 8
          flagged cases is a real stroke.
        </p>
      </div>

      <div className="md:col-span-2 rounded-card p-7 border border-accent/25 bg-gradient-to-br from-accent/18 to-accent-deep/10">
        <FontAwesomeIcon icon={faTriangleExclamation} className="text-accent text-xl" />
        <h3 className="mt-5 text-lg font-semibold">What this is not</h3>
        <p className="mt-2.5 text-fog-400 leading-relaxed max-w-[52ch]">
          A screening estimate from ten self-reported answers, with no access to your history,
          bloodwork, or imaging. Most of its accuracy comes from ranking by age: among people of
          a similar age it is much weaker, 0.66 for ages 60 to 80 and
          0.50 over 80, where 0.50 is a coin flip. It cannot diagnose
          anything.
        </p>
      </div>
    </div>
  </div>
);

export default Insights;
