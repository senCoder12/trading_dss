import { useState, useCallback, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from 'react-router-dom';

import Header from './components/layout/Header';
import Sidebar from './components/layout/Sidebar';
import StatusBar from './components/layout/StatusBar';
import NotificationBanner from './components/common/NotificationBanner';
import QuickSettings from './components/settings/QuickSettings';
import KeyboardHelp from './components/common/KeyboardHelp';

import DashboardPage from './components/dashboard/DashboardPage';
import SignalsPage from './components/signals/SignalsPage';
import PortfolioPage from './components/portfolio/PortfolioPage';
import MarketPage from './components/market/MarketPage';
import NewsPage from './components/news/NewsPage';
import AnomalyPage from './components/anomalies/AnomalyPage';
import SystemPage from './components/system/SystemPage';

import { useWebSocket } from './hooks/useWebSocket';
import { useNotification } from './hooks/useNotification';
import { DataStoreProvider } from './hooks/useDataStore';

const SHORTCUTS = [
  { key: '1', label: 'Dashboard' },
  { key: '2', label: 'Signals' },
  { key: '3', label: 'Portfolio' },
  { key: '4', label: 'Market' },
  { key: '5', label: 'News' },
  { key: '6', label: 'Anomalies' },
  { key: '7', label: 'System' },
  { key: 'S', label: 'Settings' },
  { key: 'Esc', label: 'Close panel' },
  { key: '?', label: 'This help' },
];

function KeyboardShortcuts({ setSettingsOpen }) {
  const navigate = useNavigate();
  const [showHelp, setShowHelp] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e) => {
      // Don't trigger in input fields
      const tag = e.target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      switch (e.key) {
        case '1': navigate('/'); break;
        case '2': navigate('/signals'); break;
        case '3': navigate('/portfolio'); break;
        case '4': navigate('/market'); break;
        case '5': navigate('/news'); break;
        case '6': navigate('/anomalies'); break;
        case '7': navigate('/system'); break;
        case 's': setSettingsOpen(true); break;
        case 'Escape':
          setSettingsOpen(false);
          setShowHelp(false);
          break;
        case '?': setShowHelp((prev) => !prev); break;
        default: return;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [navigate, setSettingsOpen]);

  return <KeyboardHelp shortcuts={SHORTCUTS} isOpen={showHelp} onClose={() => setShowHelp(false)} />;
}

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [toasts, setToasts] = useState([]);
  const { notify } = useNotification();

  const addToast = useCallback((toast) => {
    const id = Date.now() + Math.random();
    setToasts((prev) => [...prev, { ...toast, id }]);
    // Auto-dismiss after 30 seconds
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 30_000);
  }, []);

  const handleWsMessage = useCallback((message) => {
    if (message.type === 'signal') {
      const { index_id, signal_type, confidence, entry } = message.data;
      const emoji = signal_type === 'BUY_CALL' ? '\u{1F7E2}' : '\u{1F534}';
      const title = `${emoji} ${signal_type} \u2014 ${index_id}`;
      const body = `Confidence: ${confidence} | Entry: ${Number(entry).toLocaleString('en-IN')}`;

      notify({ title, body, priority: 'CRITICAL', sound: true });
      addToast({
        type: 'signal',
        message: `${title}\n${body}`,
        priority: 'CRITICAL',
        timestamp: message.timestamp,
      });
    }

    if (message.type === 'position_exit') {
      const { index_id, outcome, pnl } = message.data;
      const emoji = outcome === 'WIN' ? '\u2705' : '\u274C';
      const title = `${emoji} Position Closed \u2014 ${index_id}`;
      const body = `P&L: \u20B9${Number(pnl).toLocaleString('en-IN')}`;

      notify({ title, body, priority: 'HIGH', sound: true });
      addToast({
        type: 'position_exit',
        message: `${title}\n${body}`,
        priority: 'HIGH',
        timestamp: message.timestamp,
      });
    }

    if (message.type === 'system_alert') {
      const { message: msg, severity } = message.data;
      const priority = severity === 'CRITICAL' ? 'CRITICAL' : 'HIGH';
      notify({ title: 'System Alert', body: msg, priority, sound: true });
      addToast({
        type: 'system_alert',
        message: msg,
        priority,
        timestamp: message.timestamp,
      });
    }
  }, [notify, addToast]);

  const { connected } = useWebSocket(handleWsMessage);

  return (
    <BrowserRouter>
      <div className="flex h-screen bg-slate-900 text-slate-100 overflow-hidden">
        <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

        <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
          <Header
            onMenuClick={() => setSidebarOpen(true)}
            wsConnected={connected}
            onSettingsClick={() => setSettingsOpen(true)}
          />

          <main className="flex-1 overflow-y-auto">
            <DataStoreProvider>
              <Routes>
                <Route path="/" element={<DashboardPage />} />
                <Route path="/signals" element={<SignalsPage />} />
                <Route path="/portfolio" element={<PortfolioPage />} />
                <Route path="/market" element={<MarketPage />} />
                <Route path="/news" element={<NewsPage />} />
                <Route path="/anomalies" element={<AnomalyPage />} />
                <Route path="/system" element={<SystemPage />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </DataStoreProvider>
          </main>

          <StatusBar />
        </div>

        <NotificationBanner
          notifications={toasts}
          onDismiss={(id) => setToasts((prev) => prev.filter((t) => t.id !== id))}
        />

        <QuickSettings isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
        <KeyboardShortcuts setSettingsOpen={setSettingsOpen} />
      </div>
    </BrowserRouter>
  );
}
