import { X, Crosshair, LogOut, ShieldAlert, Bell } from 'lucide-react';

const TYPE_META = {
  signal: { label: 'New signal', icon: Crosshair },
  position_exit: { label: 'Position closed', icon: LogOut },
  system_alert: { label: 'System alert', icon: ShieldAlert },
};

const PRIORITY_ACCENT = {
  CRITICAL: 'text-down bg-down/10',
  HIGH: 'text-warn bg-warn/10',
  NORMAL: 'text-brand bg-brand/10',
};

/**
 * Toast stack, top-right.
 * @param {{ notifications: Array<{id, type, message, priority, timestamp}>, onDismiss: (id) => void }}
 */
export default function NotificationBanner({ notifications, onDismiss }) {
  if (!notifications.length) return null;

  return (
    <div className="fixed top-16 right-4 z-50 flex flex-col gap-2 w-[360px] max-w-[calc(100vw-2rem)]">
      {notifications.map((n) => {
        const meta = TYPE_META[n.type] ?? { label: 'Alert', icon: Bell };
        const Icon = meta.icon;
        const accent = PRIORITY_ACCENT[n.priority] ?? PRIORITY_ACCENT.NORMAL;
        return (
          <div key={n.id} className="card shadow-pop p-3.5 flex items-start gap-3 animate-slide-in">
            <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${accent}`}>
              <Icon className="w-4 h-4" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold text-ink">{meta.label}</span>
                <span className="text-2xs text-ink-faint num">
                  {new Date(n.timestamp).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
              <p className="text-[13px] text-ink-muted mt-1 whitespace-pre-line break-words leading-snug">
                {n.message}
              </p>
            </div>
            <button onClick={() => onDismiss(n.id)} className="icon-btn w-6 h-6 -mr-1 -mt-1" aria-label="Dismiss">
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        );
      })}
    </div>
  );
}
