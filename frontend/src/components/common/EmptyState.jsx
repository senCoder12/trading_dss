import { Inbox } from 'lucide-react';

export function EmptyState({ icon: Icon = Inbox, title = 'No data', message, action, className = '' }) {
  return (
    <div className={`flex flex-col items-center justify-center text-center py-12 px-6 ${className}`}>
      <div className="w-10 h-10 rounded-xl bg-surface-2 border border-line flex items-center justify-center mb-3">
        <Icon className="w-5 h-5 text-ink-faint" />
      </div>
      <p className="text-sm font-medium text-ink-muted">{title}</p>
      {message && <p className="text-xs text-ink-faint mt-1 max-w-xs leading-relaxed">{message}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
