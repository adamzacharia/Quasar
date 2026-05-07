"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";

interface Props {
  children?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught React error caught by ErrorBoundary:", error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center h-full w-full min-h-[400px] text-slate-300 bg-bg-dark z-50 relative p-6">
          <div className="bg-slate-800/50 p-8 rounded-2xl border border-slate-700/50 max-w-md w-full text-center shadow-xl backdrop-blur-md">
            <h2 className="text-xl font-semibold mb-3 text-white">Interface Recovered</h2>
            <p className="text-sm text-slate-400 mb-6">
              The layout engine experienced an unexpected glitch (likely due to resizing across monitors). The application has protected your data, but needs to refresh the view to continue.
            </p>
            <button
              className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors font-medium w-full"
              onClick={() => {
                this.setState({ hasError: false });
                window.location.reload();
              }}
            >
              Refresh Interface
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
