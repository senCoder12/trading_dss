import {
  LayoutDashboard, Crosshair, Briefcase, CandlestickChart, Layers,
  Sparkles, Newspaper, BellRing, Activity,
} from 'lucide-react';

/**
 * Single source of truth for navigation: sidebar groups, header titles and
 * keyboard shortcuts all read from here.
 */
export const NAV_GROUPS = [
  {
    label: 'Overview',
    items: [
      { to: '/', end: true, label: 'Dashboard', icon: LayoutDashboard, key: '1',
        desc: 'Market snapshot, live signals and portfolio at a glance' },
    ],
  },
  {
    label: 'Trading',
    items: [
      { to: '/signals', label: 'Signals', icon: Crosshair, key: '2',
        desc: 'Model signals, history and hit-rate' },
      { to: '/portfolio', label: 'Portfolio', icon: Briefcase, key: '3',
        desc: 'Capital, open positions and trade log' },
    ],
  },
  {
    label: 'Markets',
    items: [
      { to: '/market', label: 'Markets', icon: CandlestickChart, key: '4',
        desc: 'All indices, charts and open interest' },
      { to: '/options', label: 'Option Chain', icon: Layers, key: '8',
        desc: 'Strike-wise OI, IV and buildup' },
      { to: '/universe', label: 'F&O Universe', icon: Sparkles, key: '9',
        desc: 'F&O stocks ranked by tradability' },
    ],
  },
  {
    label: 'Intelligence',
    items: [
      { to: '/news', label: 'News', icon: Newspaper, key: '5',
        desc: 'Market news and sentiment' },
      { to: '/anomalies', label: 'Alerts', icon: BellRing, key: '6',
        desc: 'Detected market anomalies' },
    ],
  },
  {
    label: 'System',
    items: [
      { to: '/system', label: 'System', icon: Activity, key: '7',
        desc: 'Health, data freshness and kill switch' },
    ],
  },
];

export const NAV_ITEMS = NAV_GROUPS.flatMap((g) => g.items);

export function routeFor(pathname) {
  return NAV_ITEMS.find((i) => (i.end ? pathname === i.to : pathname.startsWith(i.to))) ?? NAV_ITEMS[0];
}
