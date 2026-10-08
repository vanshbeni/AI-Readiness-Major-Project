import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Nebulus — Messy data in. Model-ready out.',
  description: 'Profile, score, explain and clean tabular data before it reaches your model. Reproducible pipelines and honest benchmarks.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
