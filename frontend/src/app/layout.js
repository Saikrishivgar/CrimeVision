import './globals.css';

export const metadata = {
  title: 'CrimeVision',
  description: 'YOLO surveillance analysis',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className="h-full bg-slate-950 text-slate-100">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet" />
      </head>
      <body className="h-full font-sans" style={{ fontFamily: 'Inter, sans-serif' }}>
        {children}
      </body>
    </html>
  );
}
