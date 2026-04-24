import { useEffect, useRef, useState } from 'react';

/**
 * Renders a price value that flashes green on up-tick and red on down-tick,
 * mimicking the live price behaviour seen in broker trading apps.
 */
export function FlashPrice({ value, formatter = String, className = '' }) {
  const prevRef = useRef(value);
  const [animKey, setAnimKey] = useState(0);
  const [dir, setDir] = useState(null); // 'up' | 'down' | null

  useEffect(() => {
    if (value == null) return;
    const prev = prevRef.current;
    if (prev != null && value !== prev) {
      setDir(value > prev ? 'up' : 'down');
      setAnimKey((k) => k + 1); // restart animation on every tick
    }
    prevRef.current = value;
  }, [value]);

  return (
    <span
      key={animKey}
      className={`rounded-sm px-0.5 ${className}`}
      style={dir ? { animation: `flash-${dir} 800ms ease-out forwards` } : undefined}
    >
      {formatter(value)}
    </span>
  );
}
