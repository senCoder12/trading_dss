import { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { api } from '../../api/client';

const INDICES = ['NIFTY50', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'];

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

  useEffect(() => {
    if (!isOpen) return;
    api.getLiveConfig().then((config) => {
      if (config) setSettings((prev) => ({ ...prev, ...config }));
    }).catch(() => {});
  }, [isOpen]);

  const handleSave = async () => {
    setSaving(true);
    try {
      await api.updateLiveConfig(settings);
      onClose();
    } catch (err) {
      console.error('Failed to save settings:', err);
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
    <div className="fixed inset-0 z-40">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />

      {/* Panel */}
      <div className="absolute right-0 top-0 h-full w-80 bg-slate-800 border-l border-slate-700 p-6 overflow-y-auto animate-slide-in">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-lg font-semibold text-slate-100">Quick Settings</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-200">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Risk Per Trade */}
        <div className="mb-5">
          <label className="text-xs text-slate-400 font-medium">Risk Per Trade</label>
          <input
            type="range"
            min="0.5"
            max="5"
            step="0.5"
            value={settings.riskPerTrade}
            onChange={(e) => set('riskPerTrade', parseFloat(e.target.value))}
            className="w-full mt-1.5 accent-blue-500"
          />
          <span className="text-slate-300 text-sm font-mono">{settings.riskPerTrade}%</span>
        </div>

        {/* Min Confidence */}
        <div className="mb-5">
          <label className="text-xs text-slate-400 font-medium">Min Confidence</label>
          <select
            value={settings.minConfidence}
            onChange={(e) => set('minConfidence', e.target.value)}
            className="w-full mt-1.5 bg-slate-700 text-slate-200 rounded p-2 text-sm border border-slate-600 focus:border-blue-500 outline-none"
          >
            <option value="LOW">LOW (more trades)</option>
            <option value="MEDIUM">MEDIUM (balanced)</option>
            <option value="HIGH">HIGH (fewer trades)</option>
          </select>
        </div>

        {/* Max Positions */}
        <div className="mb-5">
          <label className="text-xs text-slate-400 font-medium">Max Positions</label>
          <select
            value={settings.maxPositions}
            onChange={(e) => set('maxPositions', parseInt(e.target.value))}
            className="w-full mt-1.5 bg-slate-700 text-slate-200 rounded p-2 text-sm border border-slate-600 focus:border-blue-500 outline-none"
          >
            {[1, 2, 3, 4, 5].map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </div>

        {/* Active Indices */}
        <div className="mb-5">
          <label className="text-xs text-slate-400 font-medium">Active Indices</label>
          <div className="mt-1.5 space-y-1.5">
            {INDICES.map((idx) => (
              <label key={idx} className="flex items-center gap-2 text-slate-300 text-sm cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.activeIndices.includes(idx)}
                  onChange={() => toggleIndex(idx)}
                  className="rounded accent-blue-500"
                />
                {idx}
              </label>
            ))}
          </div>
        </div>

        {/* Sound Alerts */}
        <div className="mb-5">
          <label className="flex items-center gap-2 text-slate-300 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={settings.soundEnabled}
              onChange={() => set('soundEnabled', !settings.soundEnabled)}
              className="rounded accent-blue-500"
            />
            Sound Alerts
          </label>
        </div>

        {/* Notification Threshold */}
        <div className="mb-6">
          <label className="text-xs text-slate-400 font-medium">Notification Threshold</label>
          <select
            value={settings.notificationThreshold}
            onChange={(e) => set('notificationThreshold', e.target.value)}
            className="w-full mt-1.5 bg-slate-700 text-slate-200 rounded p-2 text-sm border border-slate-600 focus:border-blue-500 outline-none"
          >
            <option value="ALL">ALL</option>
            <option value="HIGH">HIGH+ only</option>
            <option value="CRITICAL">CRITICAL only</option>
          </select>
        </div>

        <button
          onClick={handleSave}
          disabled={saving}
          className="w-full py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:text-slate-400 text-white text-sm font-medium rounded transition-colors"
        >
          {saving ? 'Saving...' : 'Save Settings'}
        </button>
      </div>
    </div>
  );
}
