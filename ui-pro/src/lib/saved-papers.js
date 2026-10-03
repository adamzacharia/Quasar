/**
 * Saved-paper (bookmark) identity and list merging. Pure so node tests cover it.
 *
 * A paper card's `id` is not a safe bookmark key: a paper without a bibcode
 * gets a positional id (`paper-0`, `paper-1`, ...), so bookmarking result #1
 * of one search would light up result #1 of every other search, and on the
 * server two different papers would overwrite each other. The key is built
 * from the paper's real identifiers instead, best first.
 */

function clean(value) {
    return String(value ?? "").trim();
}

/** Stable bookmark key for a paper, or "" when it has nothing to key on. */
export function savedPaperKey(paper) {
    if (!paper) return "";
    const bibcode = clean(paper.bibcode);
    if (bibcode) return bibcode;
    const doi = clean(paper.doi).toLowerCase().replace(/^https?:\/\/(dx\.)?doi\.org\//, "");
    if (doi) return `doi:${doi}`;
    const arxiv = clean(paper.arxivId).replace(/^arxiv:/i, "");
    if (arxiv) return `arxiv:${arxiv}`;
    const title = clean(paper.title).toLowerCase().replace(/\s+/g, " ");
    if (title && title !== "untitled") return `title:${title}|${Number(paper.year) || 0}`;
    return "";
}

/**
 * Merge the server's bookmark list with what is already on screen. The server
 * list is authoritative and keeps its order; a local bookmark the server does
 * not know yet (its save is still in flight) is kept at the end rather than
 * dropped, so a slow load can never undo a click.
 */
export function mergeSavedPapers(serverPapers, localPapers) {
    const merged = [];
    const seen = new Set();
    for (const paper of serverPapers || []) {
        const id = clean(paper?.id) || savedPaperKey(paper);
        if (!id || seen.has(id)) continue;
        seen.add(id);
        merged.push({ ...paper, id });
    }
    for (const paper of localPapers || []) {
        if (!paper?.id || seen.has(paper.id)) continue;
        seen.add(paper.id);
        merged.push(paper);
    }
    return merged;
}
