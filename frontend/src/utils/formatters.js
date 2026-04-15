const IN_NUM = new Intl.NumberFormat('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const IN_NUM_0 = new Intl.NumberFormat('en-IN', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
const IST = 'Asia/Kolkata';

export function formatCurrency(value, decimals = 2) {
  if (value === null || value === undefined || isNaN(value)) return '--';
  const fmt = decimals === 0 ? IN_NUM_0 : IN_NUM;
  return '₹' + fmt.format(value);
}

export function formatNumber(value, decimals = 2) {
  if (value === null || value === undefined || isNaN(value)) return '--';
  const fmt = new Intl.NumberFormat('en-IN', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
  return fmt.format(value);
}

export function formatPrice(value) {
  return formatNumber(value, 2);
}

export function formatPercentage(value, showSign = true) {
  if (value === null || value === undefined || isNaN(value)) return '--';
  const sign = showSign && value > 0 ? '+' : '';
  return `${sign}${Number(value).toFixed(2)}%`;
}

export function formatPnL(value) {
  if (value === null || value === undefined || isNaN(value)) return '--';
  const sign = value >= 0 ? '+' : '';
  return sign + formatCurrency(value);
}

export function formatLargeNumber(value) {
  if (value === null || value === undefined || isNaN(value)) return '--';
  const abs = Math.abs(value);
  if (abs >= 1_00_00_000) return (value / 1_00_00_000).toFixed(2) + ' Cr';
  if (abs >= 1_00_000) return (value / 1_00_000).toFixed(2) + ' L';
  if (abs >= 1_000) return (value / 1_000).toFixed(1) + 'K';
  return formatNumber(value, 0);
}

function getIST(date) {
  const d = date instanceof Date ? date : new Date(date);
  return new Date(d.toLocaleString('en-US', { timeZone: IST }));
}

export function formatIST(date) {
  if (!date) return '--';
  const d = date instanceof Date ? date : new Date(date);
  return d.toLocaleTimeString('en-IN', { timeZone: IST, hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
}

export function formatISTDate(date) {
  if (!date) return '--';
  const d = date instanceof Date ? date : new Date(date);
  return d.toLocaleDateString('en-IN', { timeZone: IST, day: '2-digit', month: 'short', year: 'numeric' });
}

export function formatISTDateTime(date) {
  if (!date) return '--';
  const d = date instanceof Date ? date : new Date(date);
  return d.toLocaleString('en-IN', { timeZone: IST, day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false });
}

export function timeAgo(date) {
  if (!date) return '--';
  const d = date instanceof Date ? date : new Date(date);
  const seconds = Math.floor((Date.now() - d.getTime()) / 1000);
  if (seconds < 5) return 'just now';
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

export function isMarketOpen() {
  const ist = getIST(new Date());
  const day = ist.getDay();
  if (day === 0 || day === 6) return false;
  const mins = ist.getHours() * 60 + ist.getMinutes();
  return mins >= 9 * 60 + 15 && mins <= 15 * 60 + 30;
}

export function getMarketTimeInfo() {
  const ist = getIST(new Date());
  const day = ist.getDay();
  if (day === 0 || day === 6) return { open: false, label: 'CLOSED (Weekend)' };

  const mins = ist.getHours() * 60 + ist.getMinutes();
  const openAt = 9 * 60 + 15;
  const closeAt = 15 * 60 + 30;

  if (mins < openAt) {
    const left = openAt - mins;
    return { open: false, label: `Opens in ${Math.floor(left / 60)}h ${left % 60}m` };
  }
  if (mins > closeAt) {
    return { open: false, label: 'CLOSED' };
  }
  const remaining = closeAt - mins;
  const h = Math.floor(remaining / 60);
  const m = remaining % 60;
  return {
    open: true,
    label: h > 0 ? `OPEN · ${h}h ${m}m left` : `OPEN · ${m}m left`,
  };
}

export function formatConfidence(score) {
  if (score === null || score === undefined) return '--';
  return `${Math.round(score * 100)}%`;
}
