export function normalizeEvidenceQuality(value) {
    if (!value || typeof value !== "object") return undefined;
    const score = Number(value.score);
    const normalizedScore = Number.isFinite(score) ? Math.max(0, Math.min(100, Math.round(score))) : 0;
    return {
        score: normalizedScore,
        tier: String(value.tier || "general"),
        label: String(value.label || "General web"),
        reason: String(value.reason || ""),
        signals: Array.isArray(value.signals) ? value.signals.map(String).filter(Boolean) : [],
    };
}

export function mergeEvidenceQuality(existing, incoming) {
    const existingQuality = normalizeEvidenceQuality(existing);
    const incomingQuality = normalizeEvidenceQuality(incoming);
    if (!existingQuality) return incomingQuality;
    if (!incomingQuality) return existingQuality;
    return incomingQuality.score > existingQuality.score ? incomingQuality : existingQuality;
}

export function evidenceScore(source) {
    return normalizeEvidenceQuality(source?.evidenceQuality)?.score || 0;
}

export function rankWebSources(sources = []) {
    return sources
        .map((source, index) => ({
            ...source,
            evidenceQuality: normalizeEvidenceQuality(source?.evidenceQuality),
            _rankIndex: index,
        }))
        .sort((a, b) => {
            const scoreDiff = evidenceScore(b) - evidenceScore(a);
            return scoreDiff || a._rankIndex - b._rankIndex;
        })
        .map((source) => {
            const clean = { ...source };
            delete clean._rankIndex;
            return clean;
        });
}
