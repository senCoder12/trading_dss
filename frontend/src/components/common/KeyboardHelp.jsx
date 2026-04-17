import { Keyboard } from 'lucide-react';

export default function KeyboardHelp({ shortcuts, isOpen, onClose }) {
  return (
    <>
      {/* Floating ? button in bottom-right corner */}
      <button
        onClick={onClose}
        className="fixed bottom-4 right-4 z-30 w-8 h-8 rounded-full bg-slate-700 border border-slate-600 text-slate-400 hover:text-slate-200 hover:bg-slate-600 flex items-center justify-center text-xs font-bold transition-colors"
        title="Keyboard shortcuts (?)"
      >
        <Keyboard className="w-4 h-4" />
      </button>

      {/* Shortcuts popup */}
      {isOpen && (
        <div className="fixed bottom-14 right-4 z-40 bg-slate-800 border border-slate-700 rounded-lg shadow-xl p-4 w-56">
          <h3 className="text-slate-100 text-xs font-semibold mb-3 tracking-wide uppercase">
            Keyboard Shortcuts
          </h3>
          <div className="space-y-1.5">
            {shortcuts.map(({ key, label }) => (
              <div key={key} className="flex items-center justify-between text-xs">
                <span className="text-slate-400">{label}</span>
                <kbd className="px-1.5 py-0.5 bg-slate-700 border border-slate-600 rounded text-slate-300 font-mono text-[10px]">
                  {key}
                </kbd>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
