import { useEffect, useRef, useState, useCallback } from 'react';

/**
 * Maintains a WebSocket connection to the backend with auto-reconnect.
 *
 * @param {(message: object) => void} onMessage — called for every parsed event
 * @returns {{ connected: boolean }}
 */
export function useWebSocket(onMessage) {
  const wsRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const reconnectTimeout = useRef(null);
  const onMessageRef = useRef(onMessage);

  // Keep callback ref current without triggering reconnects
  useEffect(() => { onMessageRef.current = onMessage; });

  const connect = useCallback(() => {
    // Clean up any pending reconnect
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = null;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    const ws = new WebSocket(wsUrl);
    let pingInterval = null;

    ws.onopen = () => {
      setConnected(true);
      console.log('WebSocket connected');
      // Keep-alive ping every 30 seconds
      pingInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send('ping');
      }, 30_000);
    };

    ws.onmessage = (event) => {
      if (event.data === 'pong') return;
      try {
        const message = JSON.parse(event.data);
        onMessageRef.current(message);
      } catch (e) {
        console.error('WebSocket message parse error:', e);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      clearInterval(pingInterval);
      console.log('WebSocket disconnected — reconnecting in 3s');
      reconnectTimeout.current = setTimeout(connect, 3000);
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      ws.close();
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
    };
  }, [connect]);

  return { connected };
}
