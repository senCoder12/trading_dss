import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * Custom hook for periodic data fetching with state management.
 * 
 * @param {Function} fetchFn - Async function that returns data. 
 * @param {number} intervalMs - Polling interval in milliseconds.
 * @returns {Object} { data, loading, error, lastUpdated, isStale, refetch }
 */
export function usePolling(fetchFn, intervalMs = 10_000) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [isStale, setIsStale] = useState(false);
  
  const consecutiveErrorsRef = useRef(0);
  const activeFetchRef = useRef(0);

  const poll = useCallback(async (showLoading = false) => {
    const fetchId = ++activeFetchRef.current;
    
    if (showLoading) {
      setLoading(true);
      // Optional: don't clear old data to avoid flickering if UI handles it
    }

    try {
      const result = await fetchFn();
      
      // Only update state if this is still the most recent fetch request
      if (fetchId === activeFetchRef.current) {
        setData(result);
        setLastUpdated(Date.now());
        setError(null);
        setIsStale(false);
        consecutiveErrorsRef.current = 0;
      }
    } catch (err) {
      if (fetchId === activeFetchRef.current) {
        setError(err.message || 'Fetch failed');
        consecutiveErrorsRef.current += 1;
        
        // After 3 failures, mark as stale but keep old data visible
        if (consecutiveErrorsRef.current >= 3) {
          setIsStale(true);
        }
      }
    } finally {
      if (fetchId === activeFetchRef.current) {
        setLoading(false);
      }
    }
  }, [fetchFn]);

  // Handle dependency changes (e.g. index switch) and initial mount
  useEffect(() => {
    poll(true);
    
    const id = setInterval(() => {
      poll(false);
    }, intervalMs);

    return () => clearInterval(id);
  }, [poll, intervalMs]);

  return { 
    data, 
    loading, 
    error, 
    lastUpdated, 
    isStale, 
    refetch: () => poll(true) 
  };
}
