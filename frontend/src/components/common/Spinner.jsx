const sizes = { xs: 'w-3 h-3 border', sm: 'w-4 h-4 border', md: 'w-7 h-7 border-2', lg: 'w-12 h-12 border-2' };

export function Spinner({ size = 'md', className = '' }) {
  return (
    <div
      className={`animate-spin rounded-full border-slate-700 border-t-blue-400 ${sizes[size]} ${className}`}
    />
  );
}

export function LoadingOverlay({ message = 'Loading...' }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-12 text-slate-400">
      <Spinner size="lg" />
      <span className="text-sm">{message}</span>
    </div>
  );
}
