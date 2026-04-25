import { useState, useEffect, useCallback, useRef } from 'react';

export function usePolling(fetchFn, intervalMs = 10_000) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [isStale, setIsStale] = useState(false);

  const consecutiveErrorsRef = useRef(0);
  const activeFetchRef = useRef(0);
  // Always holds the latest fetchFn without being a dependency.
  // This breaks the inline-arrow → poll → useEffect infinite loop.
  const fetchFnRef = useRef(fetchFn);
  fetchFnRef.current = fetchFn;

  const poll = useCallback(async (showLoading = false) => {
    const fetchId = ++activeFetchRef.current;

    if (showLoading) setLoading(true);

    try {
      const result = await fetchFnRef.current();

      if (fetchId === activeFetchRef.current) {
        setData(result);
        setLastUpdated(Date.now());
        setError(null);
        consecutiveErrorsRef.current = 0;
        setIsStale(false);
      }
    } catch (err) {
      if (fetchId === activeFetchRef.current) {
        setError(err.message || 'Fetch failed');
        consecutiveErrorsRef.current += 1;
        if (consecutiveErrorsRef.current >= 3) setIsStale(true);
      }
    } finally {
      if (fetchId === activeFetchRef.current) setLoading(false);
    }
  }, []); // stable — no deps needed; fetchFn is read via ref

  useEffect(() => {
    poll(true);
    const id = setInterval(() => poll(false), intervalMs);
    return () => clearInterval(id);
  }, [poll, intervalMs]);

  return { data, loading, error, lastUpdated, isStale, refetch: () => poll(true) };
}
