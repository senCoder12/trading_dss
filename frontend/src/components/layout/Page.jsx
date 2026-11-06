/** Standard page body: centered, max-width, consistent gutters and vertical rhythm. */
export function Page({ children, className = '' }) {
  return (
    <div className={`w-full max-w-[1600px] mx-auto px-4 lg:px-6 py-5 lg:py-6 space-y-5 ${className}`}>
      {children}
    </div>
  );
}

/** Full-height page for split-pane tools (option chain, universe). */
export function FullPage({ children, className = '' }) {
  return <div className={`h-full min-h-0 flex flex-col ${className}`}>{children}</div>;
}

/** Row above page content: left = tabs/filters, right = actions. */
export function Toolbar({ children, right, className = '' }) {
  return (
    <div className={`flex flex-wrap items-center justify-between gap-3 ${className}`}>
      <div className="flex flex-wrap items-center gap-3 min-w-0">{children}</div>
      {right && <div className="flex items-center gap-2">{right}</div>}
    </div>
  );
}
