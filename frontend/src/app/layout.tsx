import type { Metadata } from "next";
import { IBM_Plex_Mono, Manrope } from "next/font/google";
import { ThemeProvider } from "next-themes";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/sonner";
import "./globals.css";

const sans = Manrope({
  subsets: ["latin"],
  variable: "--font-manrope",
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-plex",
});

export const metadata: Metadata = {
  title: "OperateSolve",
  description: "AI copilot for enterprise operations. Triage incidents, review changes, and draft status updates.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable} dark h-full`} suppressHydrationWarning>
      <body className="h-full overflow-hidden">
        <ThemeProvider attribute="class" forcedTheme="dark" enableSystem={false}>
          <TooltipProvider delay={280}>
            {children}
            <Toaster position="top-right" />
          </TooltipProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
