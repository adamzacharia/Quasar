"use client";

import { useState, useEffect } from "react";
import { useAuthStore } from "@/lib/auth-store";
import { Loader2, Mail, Lock, User as UserIcon, X, Chrome } from "lucide-react";

export function AuthModal() {
    const { isAuthModalOpen, closeAuthModal, setAuth } = useAuthStore();
    const [isLogin, setIsLogin] = useState(true);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [showEmailForm, setShowEmailForm] = useState(false);

    // Form states
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [displayName, setDisplayName] = useState("");

    // Load Google script when modal opens
    useEffect(() => {
        if (!isAuthModalOpen) return;

        const script = document.createElement("script");
        script.src = "https://accounts.google.com/gsi/client";
        script.async = true;
        script.defer = true;
        document.body.appendChild(script);

        script.onload = () => {
            if (window.google) {
                window.google.accounts.id.initialize({
                    client_id: process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID || "84870453296-jmmnt0c85sfb648hvb9sceggooutj5ee.apps.googleusercontent.com",
                    callback: handleGoogleResponse,
                });

                // Small delay to ensure the container is rendered
                setTimeout(() => {
                    const btnContainer = document.getElementById("google-signin-button");
                    if (btnContainer) {
                        window.google.accounts.id.renderButton(
                            btnContainer,
                            { theme: "outline", size: "large", width: 320, text: "continue_with" }
                        );
                    }
                }, 100);
            }
        };

        return () => {
            if (document.body.contains(script)) {
                document.body.removeChild(script);
            }
        };
    }, [isAuthModalOpen]);

    if (!isAuthModalOpen) return null;

    const handleGoogleResponse = async (response: any) => {
        setIsLoading(true);
        setError(null);
        try {
            const res = await fetch("http://localhost:8000/api/auth/google", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ credential: response.credential }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Google authentication failed");

            setAuth(data.user, data.token);
            closeAuthModal();
        } catch (err: any) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setError(null);

        const endpoint = isLogin ? "/api/auth/login" : "/api/auth/register";
        const body = isLogin
            ? { username: email, password }
            : { username: email, password, email, display_name: displayName };

        try {
            const res = await fetch(`http://localhost:8000${endpoint}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            const data = await res.json();

            if (!res.ok) throw new Error(data.detail || "Authentication failed");

            setAuth(data.user, data.token);
            closeAuthModal();
        } catch (err: any) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-md animate-in fade-in duration-300">
            {/* Click outside to close */}
            <div className="absolute inset-0" onClick={closeAuthModal} />

            {/* Modal Content */}
            <div className="relative w-full max-w-[400px] bg-[#1c1c1c] border border-slate-800 rounded-3xl shadow-2xl p-8 animate-in zoom-in-95 duration-300">
                <button
                    onClick={closeAuthModal}
                    className="absolute top-4 right-4 text-slate-500 hover:text-white transition-colors"
                >
                    <X className="w-5 h-5" />
                </button>

                <div className="text-center mb-8">
                    <h2 className="text-2xl text-white font-[400] tracking-tight mb-2 font-serif">
                        Sign in to QUASAR
                    </h2>
                    <p className="text-sm text-slate-400">
                        Unlock full access to the AI radio astronomy assistant
                    </p>
                </div>

                <div className="space-y-4">
                    {/* Google Button Container */}
                    <div className="flex justify-center w-full min-h-[44px]">
                        <div id="google-signin-button" className="w-full max-w-[320px] flex justify-center [&>div]:w-full" />
                    </div>

                    {!showEmailForm ? (
                        <>
                            <button
                                onClick={() => setShowEmailForm(true)}
                                className="w-full flex items-center justify-center gap-2 bg-[#2a2a2a] hover:bg-[#333] text-white border border-slate-700/50 py-2.5 px-4 rounded-lg transition-colors text-sm font-medium"
                            >
                                <Mail className="w-4 h-4" />
                                Continue with Email
                            </button>

                            <p className="text-center text-[11px] text-slate-500 mt-6 px-4">
                                By continuing, you agree to our Terms of Service and Privacy Policy.
                            </p>
                        </>
                    ) : (
                        <div className="animate-in fade-in slide-in-from-top-2 duration-300">
                            <div className="flex bg-slate-800/50 p-1 rounded-lg mb-6">
                                <button
                                    className={`flex-1 py-1.5 text-xs font-medium rounded-md transition-all ${isLogin ? 'bg-slate-700 text-white shadow-sm' : 'text-slate-400 hover:text-slate-300'}`}
                                    onClick={() => { setIsLogin(true); setError(null); }}
                                >
                                    Log In
                                </button>
                                <button
                                    className={`flex-1 py-1.5 text-xs font-medium rounded-md transition-all ${!isLogin ? 'bg-slate-700 text-white shadow-sm' : 'text-slate-400 hover:text-slate-300'}`}
                                    onClick={() => { setIsLogin(false); setError(null); }}
                                >
                                    Sign Up
                                </button>
                            </div>

                            <form onSubmit={handleSubmit} className="space-y-4">
                                {error && (
                                    <div className="bg-red-500/10 border border-red-500/20 text-red-400 text-xs px-3 py-2 rounded-lg text-center">
                                        {error}
                                    </div>
                                )}

                                {!isLogin && (
                                    <div className="relative">
                                        <UserIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                                        <input
                                            type="text"
                                            required
                                            value={displayName}
                                            onChange={(e) => setDisplayName(e.target.value)}
                                            className="w-full bg-[#111] border border-slate-800 text-white text-sm rounded-xl focus:ring-1 focus:ring-slate-600 focus:border-slate-600 pl-10 pr-4 py-3 outline-none transition-all placeholder:text-slate-600"
                                            placeholder="Your Name"
                                        />
                                    </div>
                                )}

                                <div className="relative">
                                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                                    <input
                                        type="email"
                                        required
                                        value={email}
                                        onChange={(e) => setEmail(e.target.value)}
                                        className="w-full bg-[#111] border border-slate-800 text-white text-sm rounded-xl focus:ring-1 focus:ring-slate-600 focus:border-slate-600 pl-10 pr-4 py-3 outline-none transition-all placeholder:text-slate-600"
                                        placeholder="Email address"
                                    />
                                </div>

                                <div className="relative">
                                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                                    <input
                                        type="password"
                                        required
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        className="w-full bg-[#111] border border-slate-800 text-white text-sm rounded-xl focus:ring-1 focus:ring-slate-600 focus:border-slate-600 pl-10 pr-4 py-3 outline-none transition-all placeholder:text-slate-600"
                                        placeholder="Password"
                                    />
                                </div>

                                <button
                                    type="submit"
                                    disabled={isLoading}
                                    className="w-full bg-white hover:bg-slate-200 text-black font-semibold text-sm py-3 px-4 rounded-xl transition-all disabled:opacity-70 flex items-center justify-center mt-2 group"
                                >
                                    {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : (isLogin ? "Sign In" : "Create Account")}
                                </button>

                                <button
                                    type="button"
                                    onClick={() => setShowEmailForm(false)}
                                    className="w-full text-center text-xs text-slate-500 hover:text-white transition-colors mt-4"
                                >
                                    Back to options
                                </button>
                            </form>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

declare global {
    interface Window {
        google?: any;
    }
}
