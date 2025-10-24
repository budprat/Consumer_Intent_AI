import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Purchase Intent Analysis",
  description: "AI-powered purchase intent testing using SSR methodology",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Providers>
          <div className="min-h-screen bg-slate-50">
            <nav className="border-b bg-white">
              <div className="container mx-auto px-4 py-4">
                <h1 className="text-2xl font-bold text-slate-900">
                  Purchase Intent Analysis
                </h1>
              </div>
            </nav>
            <main className="container mx-auto px-4 py-8">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
