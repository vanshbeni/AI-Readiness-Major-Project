import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'AegisMind | AI Data Readiness & Model Recommendation Platform',
  description: 'Explainable Pre-ML Data Diagnosis, Cleaning & Model Recommendation System',
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
