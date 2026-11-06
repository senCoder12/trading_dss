import { X } from 'lucide-react';

/** Centered modal listing keyboard shortcuts. Opened with "?" or from the sidebar. */
export default function KeyboardHelp({ shortcuts, isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-[2px]" onClick={onClose} />
      <div className="relative card shadow-pop w-full max-w-sm animate-fade-in">
        <div className="card-header">
          <h3 className="card-title">Keyboard shortcuts</h3>
          <button onClick={onClose} className="icon-btn w-7 h-7" aria-label="Close">
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="p-2">
          {shortcuts.map(({ key, label }) => (
            <div key={key} className="flex items-center justify-between px-3 py-2 rounded-lg hover:bg-surface-2 text-[13px]">
              <span className="text-ink-muted">{label}</span>
              <kbd className="min-w-[24px] h-6 px-1.5 inline-flex items-center justify-center rounded-md bg-surface-3 border border-line-strong text-ink-muted font-mono text-2xs">
                {key}
              </kbd>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
