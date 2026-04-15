import { useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';

import Header from './components/layout/Header';
import Sidebar from './components/layout/Sidebar';
import StatusBar from './components/layout/StatusBar';

import DashboardPage from './components/dashboard/DashboardPage';
import SignalsPage from './components/signals/SignalsPage';
import PortfolioPage from './components/portfolio/PortfolioPage';
import MarketPage from './components/market/MarketPage';
import NewsPage from './components/news/NewsPage';
import AnomalyPage from './components/anomalies/AnomalyPage';
import SystemPage from './components/system/SystemPage';

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <BrowserRouter>
      <div className="flex h-screen bg-slate-900 text-slate-100 overflow-hidden">
        <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

        <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
          <Header onMenuClick={() => setSidebarOpen(true)} />

          <main className="flex-1 overflow-y-auto">
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
          </main>

          <StatusBar />
        </div>
      </div>
    </BrowserRouter>
  );
}
