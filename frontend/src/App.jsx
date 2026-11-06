import { useState, useCallback, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from 'react-router-dom';

import Header from './components/layout/Header';
import Sidebar from './components/layout/Sidebar';
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
import OptionsChainPage from './components/options/OptionsChainPage';
import UniversePage from './components/universe/UniversePage';

import { useNotification } from './hooks/useNotification';
import { DataStoreProvider } from './hooks/useDataStore';
import { NAV_ITEMS } from './routes';

const SHORTCUTS = [
  ...[...NAV_ITEMS].sort((a, b) => a.key.localeCompare(b.key)).map((i) => ({ key: i.key, label: i.label })),
  { key: 'S', label: 'Quick settings' },
  { key: '[', label: 'Toggle sidebar' },
  { key: 'Esc', label: 'Close panel' },
  { key: '?', label: 'Keyboard shortcuts' },
];

const ROUTE_BY_KEY = Object.fromEntries(NAV_ITEMS.map((i) => [i.key, i.to]));

function KeyboardShortcuts({ setSettingsOpen, showHelp, setShowHelp, toggleCollapsed }) {
  const navigate = useNavigate();

  useEffect(() => {
    const handleKeyDown = (e) => {
      // Don't trigger in input fields or with modifiers
      const tag = e.target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || e.metaKey || e.ctrlKey || e.altKey) return;

      if (ROUTE_BY_KEY[e.key]) { navigate(ROUTE_BY_KEY[e.key]); return; }
      switch (e.key) {
        case 's': case 'S': setSettingsOpen(true); break;
        case '[': toggleCollapsed(); break;
        case 'Escape':
          setSettingsOpen(false);
          setShowHelp(false);
          break;
        case '?': setShowHelp((prev) => !prev); break;
        default:
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [navigate, setSettingsOpen, setShowHelp, toggleCollapsed]);

  return <KeyboardHelp shortcuts={SHORTCUTS} isOpen={showHelp} onClose={() => setShowHelp(false)} />;
}

const COLLAPSE_KEY = 'tdss.sidebarCollapsed';

function readCollapsed() {
  try { return localStorage.getItem(COLLAPSE_KEY) === '1'; } catch { return false; }
}

// Inner layout — inside DataStoreProvider so header/sidebar can read shared data
function AppLayout({ sidebarOpen, setSidebarOpen, settingsOpen, setSettingsOpen, toasts, onDismissToast }) {
  const [collapsed, setCollapsed] = useState(readCollapsed);
  const [showHelp, setShowHelp] = useState(false);

  const toggleCollapsed = useCallback(() => {
    setCollapsed((c) => {
      const next = !c;
      try { localStorage.setItem(COLLAPSE_KEY, next ? '1' : '0'); } catch { /* storage unavailable */ }
      return next;
    });
  }, []);

  return (
    <div className="flex h-screen bg-canvas text-ink overflow-hidden">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        collapsed={collapsed}
        onToggleCollapsed={toggleCollapsed}
        onShowShortcuts={() => setShowHelp(true)}
      />

      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <Header
          onMenuClick={() => setSidebarOpen(true)}
          onSettingsClick={() => setSettingsOpen(true)}
        />

        <main className="flex-1 min-h-0 overflow-y-auto">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/signals" element={<SignalsPage />} />
            <Route path="/portfolio" element={<PortfolioPage />} />
            <Route path="/market" element={<MarketPage />} />
            <Route path="/news" element={<NewsPage />} />
            <Route path="/anomalies" element={<AnomalyPage />} />
            <Route path="/system" element={<SystemPage />} />
            <Route path="/options" element={<OptionsChainPage />} />
            <Route path="/universe" element={<UniversePage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>

      <NotificationBanner notifications={toasts} onDismiss={onDismissToast} />

      <QuickSettings isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
      <KeyboardShortcuts
        setSettingsOpen={setSettingsOpen}
        showHelp={showHelp}
        setShowHelp={setShowHelp}
        toggleCollapsed={toggleCollapsed}
      />
    </div>
  );
}

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [toasts, setToasts] = useState([]);
  const { notify } = useNotification();

  const addToast = useCallback((toast) => {
    const id = Date.now() + Math.random();
    setToasts((prev) => [...prev, { ...toast, id }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 30_000);
  }, []);

  const onDismissToast = useCallback((id) => setToasts((prev) => prev.filter((t) => t.id !== id)), []);

  // Handles non-price WS events forwarded from DataStoreProvider
  const handleExternalEvent = useCallback((message) => {
    if (message.type === 'signal') {
      const { index_id, signal_type, confidence, entry } = message.data;
      const title = `${signal_type === 'BUY_CALL' ? 'Buy Call' : signal_type === 'BUY_PUT' ? 'Buy Put' : signal_type} · ${index_id}`;
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
      const title = `${index_id} closed · ${outcome === 'WIN' ? 'Win' : 'Loss'}`;
      const body = `P&L: ₹${Number(pnl).toLocaleString('en-IN')}`;

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

  return (
    <BrowserRouter>
      <DataStoreProvider onExternalEvent={handleExternalEvent}>
        <AppLayout
          sidebarOpen={sidebarOpen}
          setSidebarOpen={setSidebarOpen}
          settingsOpen={settingsOpen}
          setSettingsOpen={setSettingsOpen}
          toasts={toasts}
          onDismissToast={onDismissToast}
        />
      </DataStoreProvider>
    </BrowserRouter>
  );
}
