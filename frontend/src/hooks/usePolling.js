import { useState, useEffect, useRef } from 'react';

export function usePolling(fetchFn, intervalMs = 10_000) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const fetchRef = useRef(fetchFn);

  // Keep ref current without re-triggering the effect
  useEffect(() => { fetchRef.current = fetchFn; });

  useEffect(() => {
    let active = true;

    const poll = async () => {
      try {
        const result = await fetchRef.current();
        if (active) {
          setData(result);
          setLastUpdated(new Date());
          setError(null);
        }
      } catch (err) {
        if (active) setError(err.message || 'Fetch failed');
      } finally {
        if (active) setLoading(false);
      }
    };

    poll();
    const id = setInterval(poll, intervalMs);
    return () => { active = false; clearInterval(id); };
  }, [intervalMs]);

  return { data, loading, error, lastUpdated };
}
