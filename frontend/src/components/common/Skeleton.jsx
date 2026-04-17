export function Skeleton({ width = '100%', height = '1rem', className = '' }) {
  return (
    <div
      className={`bg-slate-700 rounded animate-pulse ${className}`}
      style={{ width, height }}
    />
  );
}

export function CardSkeleton() {
  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <Skeleton width="40%" height="0.75rem" className="mb-3" />
      <Skeleton width="70%" height="1.5rem" className="mb-2" />
      <Skeleton width="55%" height="0.75rem" />
    </div>
  );
}

export function ChartSkeleton({ height = 400 }) {
  return (
    <div
      className="bg-slate-800 rounded-lg border border-slate-700 p-4"
      style={{ height }}
    >
      <div className="flex gap-2 mb-4">
        <Skeleton width="40px" height="24px" />
        <Skeleton width="40px" height="24px" />
        <Skeleton width="40px" height="24px" />
      </div>
      <Skeleton width="100%" height={`${height - 80}px`} />
    </div>
  );
}

export function SignalCardSkeleton() {
  return (
    <div className="p-3 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Skeleton width="16px" height="16px" className="rounded-full" />
          <Skeleton width="80px" height="14px" />
          <Skeleton width="60px" height="18px" className="rounded-full" />
        </div>
        <Skeleton width="50px" height="18px" className="rounded-full" />
      </div>
      <Skeleton width="90%" height="10px" />
    </div>
  );
}
