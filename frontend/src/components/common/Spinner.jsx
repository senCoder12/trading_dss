const sizes = { xs: 'w-3 h-3 border', sm: 'w-4 h-4 border-2', md: 'w-6 h-6 border-2', lg: 'w-9 h-9 border-2' };

export function Spinner({ size = 'md', className = '' }) {
  return (
    <div
      role="status"
      aria-label="Loading"
      className={`animate-spin rounded-full border-line-strong border-t-brand ${sizes[size]} ${className}`}
    />
  );
}

export function LoadingOverlay({ message = 'Loading…' }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-ink-faint">
      <Spinner size="md" />
      <span className="text-xs">{message}</span>
    </div>
  );
}
