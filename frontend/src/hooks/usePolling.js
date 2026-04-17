import { useState, useEffect, useRef } from 'react';

export function usePolling(fetchFn, intervalMs = 10_000) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [isStale, setIsStale] = useState(false);
  const fetchRef = useRef(fetchFn);
  const consecutiveErrorsRef = useRef(0);

  // Keep ref current without re-triggering the effect
  useEffect(() => { fetchRef.current = fetchFn; });

  useEffect(() => {
    let active = true;

    const poll = async () => {
      try {
        const result = await fetchRef.current();
        if (active) {
          setData(result);
          setLastUpdated(Date.now());
          setError(null);
          setIsStale(false);
          consecutiveErrorsRef.current = 0;
        }
      } catch (err) {
        if (active) {
          setError(err.message || 'Fetch failed');
          consecutiveErrorsRef.current += 1;
          if (consecutiveErrorsRef.current >= 3) {
            setIsStale(true);
            // Keep showing last known good data — don't clear setData
          }
        }
      } finally {
        if (active) setLoading(false);
      }
    };

    poll();
    const id = setInterval(poll, intervalMs);
    return () => { active = false; clearInterval(id); };
  }, [intervalMs]);

  return { data, loading, error, lastUpdated, isStale };
}
