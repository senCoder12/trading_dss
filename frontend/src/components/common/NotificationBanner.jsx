import { X } from 'lucide-react';

const TYPE_LABELS = {
  signal: 'New Signal',
  position_exit: 'Position Exit',
  system_alert: 'System Alert',
};

const PRIORITY_STYLES = {
  CRITICAL: 'bg-red-900/90 border-red-500',
  HIGH: 'bg-yellow-900/90 border-yellow-500',
  NORMAL: 'bg-slate-800/90 border-slate-600',
};

const PRIORITY_COLORS = {
  CRITICAL: 'text-red-400',
  HIGH: 'text-yellow-400',
  NORMAL: 'text-slate-400',
};

/**
 * Slide-in toast notifications anchored to the top-right of the viewport.
 *
 * @param {{ notifications: Array<{id, type, message, priority, timestamp}>, onDismiss: (id) => void }}
 */
export default function NotificationBanner({ notifications, onDismiss }) {
  if (!notifications.length) return null;

  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-md">
      {notifications.map((n) => (
        <div
          key={n.id}
          className={`p-4 rounded-lg shadow-lg border animate-slide-in ${
            PRIORITY_STYLES[n.priority] || PRIORITY_STYLES.NORMAL
          }`}
        >
          <div className="flex justify-between items-start gap-2">
            <div className="min-w-0">
              <span
                className={`text-xs font-semibold uppercase tracking-wide ${
                  PRIORITY_COLORS[n.priority] || PRIORITY_COLORS.NORMAL
                }`}
              >
                {TYPE_LABELS[n.type] || 'Alert'}
              </span>
              <p className="text-slate-100 text-sm mt-1 whitespace-pre-line break-words">
                {n.message}
              </p>
              <span className="text-slate-500 text-xs">
                {new Date(n.timestamp).toLocaleTimeString('en-IN')}
              </span>
            </div>
            <button
              onClick={() => onDismiss(n.id)}
              className="text-slate-500 hover:text-slate-300 flex-shrink-0"
              aria-label="Dismiss"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
