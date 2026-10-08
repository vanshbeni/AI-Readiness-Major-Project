import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Nebulus / dashboard',
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
