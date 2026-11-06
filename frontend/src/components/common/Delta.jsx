import { ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';
import { FlashPrice } from './FlashPrice';
import { formatPercentage } from '../../utils/formatters';

/** Coloured percentage change with direction arrow. */
export function Delta({ value, flash = false, className = '', showIcon = true }) {
  const pos = value > 0;
  const neg = value < 0;
  const color = pos ? 'text-up' : neg ? 'text-down' : 'text-ink-faint';
  const Icon = pos ? ArrowUpRight : neg ? ArrowDownRight : Minus;
  return (
    <span className={`inline-flex items-center gap-0.5 num ${color} ${className}`}>
      {showIcon && <Icon className="w-3.5 h-3.5 flex-shrink-0" />}
      {flash ? <FlashPrice value={value} formatter={formatPercentage} /> : formatPercentage(value)}
    </span>
  );
}
