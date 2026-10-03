/** Supporting-organization logos (TACC, CosmicAI, NRAO), shown at the foot of
 * the Help and Terms pages. Each logo ships a light and a dark variant
 * (public/logos/*-logo-{light,dark}.png); globals.css shows the one that
 * matches the active data-theme. */

const PARTNERS = [
    // Heights differ so the wordmarks read at a similar size (CosmicAI's arc is tall).
    { key: "tacc", name: "Texas Advanced Computing Center", href: "https://tacc.utexas.edu/", size: "h-10 md:h-11" },
    { key: "cosmicai", name: "NSF-Simons AI Institute for Cosmic Origins (CosmicAI)", href: "https://www.cosmicai.org/", size: "h-16 md:h-[72px]" },
    { key: "nrao", name: "National Radio Astronomy Observatory", href: "https://public.nrao.edu/", size: "h-10 md:h-11" },
] as const;

export function PartnerLogos({ className = "" }: { className?: string }) {
    return (
        <div className={`flex flex-wrap items-center justify-center gap-x-12 gap-y-6 ${className}`}>
            {PARTNERS.map((p) => (
                <a
                    key={p.key}
                    href={p.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    title={p.name}
                    className="opacity-80 transition-opacity hover:opacity-100"
                >
                    <img src={`/logos/${p.key}-logo-light.png`} alt={p.name} className={`q-logo-light w-auto ${p.size}`} />
                    <img src={`/logos/${p.key}-logo-dark.png`} alt={p.name} className={`q-logo-dark w-auto ${p.size}`} />
                </a>
            ))}
        </div>
    );
}
