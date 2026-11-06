import { useState, useEffect } from 'react';
import { X, Check } from 'lucide-react';
import { api } from '../../api/client';
import { Segmented } from '../common/Segmented';

const INDICES = ['NIFTY50', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'];

const CONFIDENCE_HELP = {
  LOW: 'More signals, lower average quality.',
  MEDIUM: 'Balanced — the default for most sessions.',
  HIGH: 'Only the strongest setups. Fewer trades.',
};

function Section({ title, children }) {
  return (
    <section className="space-y-4">
      <h3 className="label">{title}</h3>
      {children}
    </section>
  );
}

function Field({ label, hint, right, children }) {
  return (
    <div>
      <div className="flex items-center justify-between gap-3 mb-2">
        <span className="text-[13px] font-medium text-ink">{label}</span>
        {right}
      </div>
      {children}
      {hint && <p className="text-xs text-ink-faint mt-1.5">{hint}</p>}
    </div>
  );
}

function Switch({ checked, onChange, label }) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-5 w-9 flex-shrink-0 items-center rounded-full transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-brand/40 ${
        checked ? 'bg-brand' : 'bg-surface-3 ring-1 ring-inset ring-line-strong'
      }`}
    >
      <span className={`inline-block h-4 w-4 rounded-full bg-white shadow transition-transform ${checked ? 'translate-x-[18px]' : 'translate-x-0.5'}`} />
    </button>
  );
}

export default function QuickSettings({ isOpen, onClose }) {
  const [settings, setSettings] = useState({
    riskPerTrade: 2.0,
    minConfidence: 'MEDIUM',
    maxPositions: 3,
    activeIndices: [...INDICES],
    soundEnabled: true,
    notificationThreshold: 'HIGH',
  });
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(null);

  useEffect(() => {
    if (!isOpen) return;
    setSaveError(null);
    api.getLiveConfig().then((config) => {
      if (config) setSettings((prev) => ({ ...prev, ...config }));
    }).catch(() => {});
  }, [isOpen]);

  const handleSave = async () => {
    setSaving(true);
    setSaveError(null);
    try {
      await api.updateLiveConfig(settings);
      onClose();
    } catch (err) {
      console.error('Failed to save settings:', err);
      setSaveError(err?.message || 'Could not save settings. Try again.');
    } finally {
      setSaving(false);
    }
  };

  if (!isOpen) return null;

  const set = (key, value) => setSettings((prev) => ({ ...prev, [key]: value }));

  const toggleIndex = (idx) => {
    const next = settings.activeIndices.includes(idx)
      ? settings.activeIndices.filter((i) => i !== idx)
      : [...settings.activeIndices, idx];
    set('activeIndices', next);
  };

  return (
    <div className="fixed inset-0 z-50">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />

      <aside
        role="dialog"
        aria-label="Quick settings"
        className="absolute right-0 top-0 h-full w-[380px] max-w-full bg-panel border-l border-line shadow-pop flex flex-col animate-drawer-in"
      >
        {/* Header */}
        <header className="flex items-start justify-between gap-3 px-6 py-5 border-b border-line">
          <div>
            <h2 className="text-[15px] font-semibold text-ink tracking-tight">Quick settings</h2>
            <p className="text-xs text-ink-faint mt-0.5">Applies to the live engine</p>
          </div>
          <button onClick={onClose} className="icon-btn -mr-2 -mt-1" aria-label="Close settings">
            <X className="w-4 h-4" />
          </button>
        </header>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-8">
          <Section title="Risk">
            <Field
              label="Risk per trade"
              hint="Share of capital at risk on a single position."
              right={
                <span className="h-6 px-2 rounded-md bg-surface-3 text-xs font-semibold text-ink num inline-flex items-center">
                  {Number(settings.riskPerTrade).toFixed(1)}%
                </span>
              }
            >
              <input
                type="range"
                min="0.5"
                max="5"
                step="0.5"
                value={settings.riskPerTrade}
                onChange={(e) => set('riskPerTrade', parseFloat(e.target.value))}
                className="w-full accent-brand cursor-pointer"
                aria-label="Risk per trade"
              />
              <div className="flex justify-between text-2xs text-ink-ghost num mt-1">
                <span>0.5%</span><span>5%</span>
              </div>
            </Field>

            <Field label="Max open positions">
              <Segmented
                size="xs"
                ariaLabel="Max open positions"
                options={[1, 2, 3, 4, 5]}
                value={Number(settings.maxPositions)}
                onChange={(v) => set('maxPositions', v)}
              />
            </Field>
          </Section>

          <div className="border-t border-line" />

          <Section title="Signals">
            <Field label="Minimum confidence" hint={CONFIDENCE_HELP[settings.minConfidence]}>
              <Segmented
                size="xs"
                ariaLabel="Minimum confidence"
                options={[
                  { value: 'LOW', label: 'Low' },
                  { value: 'MEDIUM', label: 'Medium' },
                  { value: 'HIGH', label: 'High' },
                ]}
                value={settings.minConfidence}
                onChange={(v) => set('minConfidence', v)}
              />
            </Field>

            <Field
              label="Active indices"
              right={<span className="text-2xs text-ink-faint num">{settings.activeIndices.length}/{INDICES.length}</span>}
            >
              <div className="flex flex-wrap gap-2">
                {INDICES.map((idx) => {
                  const on = settings.activeIndices.includes(idx);
                  return (
                    <button
                      key={idx}
                      onClick={() => toggleIndex(idx)}
                      aria-pressed={on}
                      className={`inline-flex items-center gap-1.5 h-8 px-3 rounded-lg text-xs font-medium border transition-colors ${
                        on
                          ? 'bg-brand/10 border-brand/40 text-brand'
                          : 'border-line text-ink-faint hover:text-ink-muted hover:border-line-strong'
                      }`}
                    >
                      {on && <Check className="w-3.5 h-3.5" />}
                      {idx}
                    </button>
                  );
                })}
              </div>
            </Field>
          </Section>

          <div className="border-t border-line" />

          <Section title="Notifications">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-[13px] font-medium text-ink">Sound alerts</div>
                <div className="text-xs text-ink-faint mt-0.5">Play a chime on new signals and exits</div>
              </div>
              <Switch
                label="Sound alerts"
                checked={!!settings.soundEnabled}
                onChange={(v) => set('soundEnabled', v)}
              />
            </div>

            <Field label="Notify me for">
              <Segmented
                size="xs"
                ariaLabel="Notification threshold"
                options={[
                  { value: 'ALL', label: 'All' },
                  { value: 'HIGH', label: 'High+' },
                  { value: 'CRITICAL', label: 'Critical' },
                ]}
                value={settings.notificationThreshold}
                onChange={(v) => set('notificationThreshold', v)}
              />
            </Field>
          </Section>
        </div>

        {/* Footer */}
        <footer className="border-t border-line px-6 py-4 bg-panel">
          {saveError && <p className="text-xs text-down mb-3">{saveError}</p>}
          <div className="flex items-center justify-end gap-2">
            <button onClick={onClose} className="btn btn-secondary">Cancel</button>
            <button onClick={handleSave} disabled={saving} className="btn btn-primary min-w-[96px]">
              {saving ? 'Saving…' : 'Save changes'}
            </button>
          </div>
        </footer>
      </aside>
    </div>
  );
}
