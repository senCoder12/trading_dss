export function Skeleton({ width = '100%', height = '1rem', className = '' }) {
  return (
    <div
      className={`bg-surface-3/70 rounded-md animate-pulse ${className}`}
      style={{ width, height }}
    />
  );
}

export function CardSkeleton() {
  return (
    <div className="card p-4">
      <Skeleton width="40%" height="0.7rem" className="mb-3" />
      <Skeleton width="65%" height="1.4rem" className="mb-2" />
      <Skeleton width="50%" height="0.7rem" />
    </div>
  );
}

export function ChartSkeleton({ height = 400 }) {
  return <Skeleton width="100%" height={`${height}px`} className="rounded-lg" />;
}

export function RowSkeleton({ rows = 4 }) {
  return (
    <div className="divide-y divide-line">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="flex items-center gap-3 px-5 py-3.5">
          <Skeleton width="28px" height="28px" className="rounded-lg" />
          <div className="flex-1 space-y-1.5">
            <Skeleton width="35%" height="0.7rem" />
            <Skeleton width="60%" height="0.6rem" />
          </div>
          <Skeleton width="56px" height="1.2rem" />
        </div>
      ))}
    </div>
  );
}

// Back-compat alias
export const SignalCardSkeleton = () => <RowSkeleton rows={1} />;
