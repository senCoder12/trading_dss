import { useEffect, useCallback, useRef } from 'react';

/**
 * Generate a short beep using the Web Audio API (no audio file needed).
 */
function playBeep(frequency = 800, duration = 200) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = ctx.createOscillator();
    const gain = ctx.createGain();
    oscillator.connect(gain);
    gain.connect(ctx.destination);
    oscillator.frequency.value = frequency;
    gain.gain.value = 0.3;
    oscillator.start();
    setTimeout(() => {
      oscillator.stop();
      ctx.close();
    }, duration);
  } catch {
    // AudioContext not available — fail silently
  }
}

/**
 * Hook for browser notifications and alert sounds.
 *
 * Requests Notification permission on mount.
 * Returns a `notify()` function that plays a beep and shows a browser
 * notification (even when the tab is in background).
 *
 * @returns {{ notify: (opts: { title, body, priority?, sound? }) => void }}
 */
export function useNotification() {
  const permissionRef = useRef(
    typeof Notification !== 'undefined' ? Notification.permission : 'denied',
  );

  useEffect(() => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission().then((p) => {
        permissionRef.current = p;
      });
    }
  }, []);

  const notify = useCallback(({ title, body, priority = 'NORMAL', sound = true }) => {
    // Sound — play for anything above LOW
    if (sound && priority !== 'LOW') {
      const freq = priority === 'CRITICAL' ? 900 : 700;
      const dur = priority === 'CRITICAL' ? 300 : 200;
      playBeep(freq, dur);
      // Double beep for CRITICAL
      if (priority === 'CRITICAL') {
        setTimeout(() => playBeep(freq, dur), 350);
      }
    }

    // Browser notification (visible even when tab is not focused)
    if ('Notification' in window && Notification.permission === 'granted') {
      const notification = new Notification(title, {
        body,
        icon: '/favicon.ico',
        tag: `trading-${Date.now()}`,
        requireInteraction: priority === 'CRITICAL',
      });

      if (priority !== 'CRITICAL') {
        setTimeout(() => notification.close(), 10_000);
      }

      notification.onclick = () => {
        window.focus();
        notification.close();
      };
    }
  }, []);

  return { notify };
}
