import { Database } from 'lucide-react';

export function EmptyState({ icon: Icon = Database, title = 'No data', message, action, className = '' }) {
  return (
    <div className={`flex flex-col items-center justify-center py-10 gap-2 text-slate-500 ${className}`}>
      <Icon className="w-9 h-9 text-slate-600" />
      <p className="text-sm font-medium text-slate-400">{title}</p>
      {message && <p className="text-xs text-center max-w-xs">{message}</p>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}
