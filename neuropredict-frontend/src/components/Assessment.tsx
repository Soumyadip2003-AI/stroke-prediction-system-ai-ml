import React, { useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faUser,
  faHeartPulse,
  faBriefcase,
  faChevronLeft,
  faChevronRight,
  faCircleExclamation,
  faSpinner,
} from '@fortawesome/free-solid-svg-icons';

interface AssessmentProps {
  onComplete: (data: any) => void;
  onLoadingChange: (loading: boolean) => void;
}

// Field names and value strings are frozen. They are the API contract and
// anything downstream that reads them.
const STEPS = [
  { title: 'Personal information', icon: faUser },
  { title: 'Health metrics', icon: faHeartPulse },
  { title: 'Lifestyle and environment', icon: faBriefcase },
];

const Field: React.FC<{ id: string; label: string; hint?: string; children: React.ReactNode }> = ({
  id,
  label,
  hint,
  children,
}) => (
  <div className="flex flex-col gap-2">
    <label htmlFor={id} className="text-sm font-medium text-fog-300">
      {label}
    </label>
    {children}
    {hint && (
      <p id={`${id}-hint`} className="text-xs text-fog-400">
        {hint}
      </p>
    )}
  </div>
);

const Assessment: React.FC<AssessmentProps> = ({ onComplete, onLoadingChange }) => {
  const [currentStep, setCurrentStep] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');
  const API_BASE =
    (process.env.REACT_APP_API_BASE as string) ||
    'https://stroke-prediction-system-ai-ml.onrender.com';

  // Defaults must not invent a risk profile. These previously described a
  // 55-year-old hypertensive smoker with elevated glucose, so anyone who
  // clicked through without editing was shown "Moderate Risk, flagged" for a
  // person who does not exist. Clinical flags now default to absent and the
  // numbers to dataset medians, so risk only rises from what you actually
  // enter.
  const [formData, setFormData] = useState({
    age: 45,
    gender: 'Female',
    ever_married: 'No',
    hypertension: 'No',
    heart_disease: 'No',
    avg_glucose_level: 92,
    bmi: 28,
    work_type: 'Private',
    residence_type: 'Urban',
    smoking_status: 'never smoked',
  });

  const handleInputChange = (field: string, value: any) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  // Range checks at the boundary. The inputs carry min/max, but a typed value
  // can still land outside them, and NaN is possible whenever a field is cleared.
  // The three numeric fields hold raw text while the user types, so an empty
  // box stays empty instead of snapping to 0. They are coerced here and again
  // when the payload is built.
  const asNumber = (value: any) => (String(value).trim() === '' ? NaN : Number(value));

  const validate = () => {
    const age = asNumber(formData.age);
    const glucose = asNumber(formData.avg_glucose_level);
    const bmi = asNumber(formData.bmi);
    if (!Number.isFinite(age) || age < 1 || age > 100) return 'Enter an age between 1 and 100.';
    if (!Number.isFinite(glucose) || glucose < 50 || glucose > 300)
      return 'Enter an average glucose level between 50 and 300 mg/dL.';
    if (!Number.isFinite(bmi) || bmi < 10 || bmi > 50) return 'Enter a BMI between 10 and 50.';
    return '';
  };

  const handleSubmit = async () => {
    const validationError = validate();
    if (validationError) {
      setError(validationError);
      return;
    }

    setError('');
    setIsSubmitting(true);
    onLoadingChange(true);

    try {
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...formData,
          age: asNumber(formData.age),
          avg_glucose_level: asNumber(formData.avg_glucose_level),
          bmi: asNumber(formData.bmi),
        }),
      });

      if (!response.ok) {
        // The API validates ranges and categories and returns 400 with a
        // specific reason. Showing "service unavailable" for that would be
        // both wrong and unactionable, so surface the real message.
        const body = await response.json().catch(() => null);
        setError(
          response.status === 400 && body?.error
            ? body.error
            : 'The prediction service is unavailable right now. Please try again in a moment.'
        );
        return;
      }

      onComplete(await response.json());
    } catch {
      setError('Could not reach the prediction service. Check your connection and try again.');
    } finally {
      setIsSubmitting(false);
      onLoadingChange(false);
    }
  };

  const step = STEPS[currentStep - 1];

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <div data-reveal className="mb-10 sm:mb-14">
        <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">Stroke risk assessment</h2>
        <p className="mt-4 text-lg text-fog-400 max-w-[58ch]">
          Ten questions across three short steps. Nothing you enter is stored.
        </p>
      </div>

      <div data-reveal className="surface rounded-card p-6 sm:p-8">
        <h3 className="text-xl sm:text-2xl font-semibold flex items-center gap-3 mb-8">
          <FontAwesomeIcon icon={step.icon} className="text-accent" />
          {step.title}
        </h3>

        {currentStep === 1 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <Field id="age" label="Age" hint="Between 1 and 100.">
              <input
                id="age"
                name="age"
                type="number"
                className="field"
                min="1"
                max="100"
                aria-describedby="age-hint"
                value={formData.age}
                onChange={(e) => handleInputChange('age', e.target.value)}
              />
            </Field>

            <Field id="gender" label="Gender">
              <select
                id="gender"
                name="gender"
                className="field"
                value={formData.gender}
                onChange={(e) => handleInputChange('gender', e.target.value)}
              >
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </Field>

            <Field id="ever_married" label="Marital status">
              <select
                id="ever_married"
                name="ever_married"
                className="field"
                value={formData.ever_married}
                onChange={(e) => handleInputChange('ever_married', e.target.value)}
              >
                <option value="No">Never married</option>
                <option value="Yes">Married</option>
              </select>
            </Field>
          </div>
        )}

        {currentStep === 2 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            {(
              [
                ['hypertension', 'Hypertension'],
                ['heart_disease', 'Heart disease'],
              ] as Array<[keyof typeof formData, string]>
            ).map(([name, legend]) => (
              <fieldset key={name} className="flex flex-col gap-2">
                <legend className="text-sm font-medium text-fog-300 mb-2">{legend}</legend>
                <div className="flex gap-3">
                  {['No', 'Yes'].map((option) => (
                    <label
                      key={option}
                      className={`flex-1 flex items-center justify-center gap-2 min-h-[48px] px-4 rounded-full border cursor-pointer transition-colors duration-200 ${
                        formData[name] === option
                          ? 'border-accent bg-accent/15 text-fog-100'
                          : 'border-white/10 text-fog-400 hover:text-fog-100 hover:border-white/25'
                      }`}
                    >
                      <input
                        type="radio"
                        name={name}
                        value={option}
                        checked={formData[name] === option}
                        onChange={(e) => handleInputChange(name, e.target.value)}
                        className="sr-only"
                      />
                      {option}
                    </label>
                  ))}
                </div>
              </fieldset>
            ))}

            <Field
              id="avg_glucose_level"
              label="Average glucose level"
              hint="Milligrams per decilitre, between 50 and 300."
            >
              <input
                id="avg_glucose_level"
                name="avg_glucose_level"
                type="number"
                className="field"
                min="50"
                max="300"
                aria-describedby="avg_glucose_level-hint"
                value={formData.avg_glucose_level}
                onChange={(e) => handleInputChange('avg_glucose_level', e.target.value)}
              />
            </Field>

            <Field id="bmi" label="Body mass index" hint="Between 10 and 50.">
              <input
                id="bmi"
                name="bmi"
                type="number"
                className="field"
                min="10"
                max="50"
                step="0.1"
                aria-describedby="bmi-hint"
                value={formData.bmi}
                onChange={(e) => handleInputChange('bmi', e.target.value)}
              />
            </Field>
          </div>
        )}

        {currentStep === 3 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <Field id="work_type" label="Work type">
              <select
                id="work_type"
                name="work_type"
                className="field"
                value={formData.work_type}
                onChange={(e) => handleInputChange('work_type', e.target.value)}
              >
                <option value="Private">Private</option>
                <option value="Self-employed">Self-employed</option>
                <option value="Children">Children</option>
                <option value="Never_worked">Never worked</option>
                <option value="Govt_job">Government job</option>
              </select>
            </Field>

            <Field id="residence_type" label="Residence type">
              <select
                id="residence_type"
                name="residence_type"
                className="field"
                value={formData.residence_type}
                onChange={(e) => handleInputChange('residence_type', e.target.value)}
              >
                <option value="Urban">Urban</option>
                <option value="Rural">Rural</option>
              </select>
            </Field>

            <div className="sm:col-span-2">
              <Field id="smoking_status" label="Smoking status">
                <select
                  id="smoking_status"
                  name="smoking_status"
                  className="field"
                  value={formData.smoking_status}
                  onChange={(e) => handleInputChange('smoking_status', e.target.value)}
                >
                  <option value="never smoked">Never smoked</option>
                  <option value="formerly smoked">Formerly smoked</option>
                  <option value="smokes">Currently smokes</option>
                  <option value="Unknown">Unknown</option>
                </select>
              </Field>
            </div>
          </div>
        )}

        {error && (
          <p
            role="alert"
            className="mt-8 flex items-start gap-2.5 text-sm text-[#f8b4c0] bg-[#f8b4c0]/10 border border-[#f8b4c0]/25 rounded-input px-4 py-3"
          >
            <FontAwesomeIcon icon={faCircleExclamation} className="mt-0.5 shrink-0" />
            {error}
          </p>
        )}

        <div className="flex items-center justify-between gap-4 mt-10">
          <button
            type="button"
            onClick={() => setCurrentStep((s) => Math.max(1, s - 1))}
            disabled={currentStep === 1 || isSubmitting}
            className="btn-quiet rounded-full px-5 py-3 min-h-[48px] flex items-center gap-2 text-sm font-medium"
          >
            <FontAwesomeIcon icon={faChevronLeft} />
            Back
          </button>

          <ol className="flex gap-2" aria-label={`Step ${currentStep} of 3`}>
            {[1, 2, 3].map((s) => (
              <li
                key={s}
                aria-current={s === currentStep ? 'step' : undefined}
                className={`h-1.5 rounded-full transition-all duration-300 ease-out-expo ${
                  s === currentStep ? 'w-8 bg-accent' : s < currentStep ? 'w-4 bg-accent/50' : 'w-4 bg-white/15'
                }`}
              />
            ))}
          </ol>

          {currentStep < 3 ? (
            <button
              type="button"
              onClick={() => setCurrentStep((s) => Math.min(3, s + 1))}
              className="btn-primary rounded-full px-6 py-3 min-h-[48px] flex items-center gap-2 text-base font-semibold whitespace-nowrap"
            >
              Next
              <FontAwesomeIcon icon={faChevronRight} />
            </button>
          ) : (
            <button
              type="button"
              onClick={handleSubmit}
              disabled={isSubmitting}
              className="btn-primary rounded-full px-6 py-3 min-h-[48px] flex items-center gap-2.5 text-base font-semibold whitespace-nowrap"
            >
              {isSubmitting && <FontAwesomeIcon icon={faSpinner} spin />}
              {isSubmitting ? 'Analysing' : 'Get my risk score'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default Assessment;
