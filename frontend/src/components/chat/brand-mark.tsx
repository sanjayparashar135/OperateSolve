export function BrandMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden="true">
      <rect x="1.2" y="1.2" width="29.6" height="29.6" rx="9" fill="currentColor" opacity="0.14" />
      <path
        d="M8.5 21.5 13.8 10.2l3.3 6.2 3.5-6.2 5.4 11.3"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
