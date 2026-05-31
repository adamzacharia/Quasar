const PROJECT_CODE_RE = /\b\d{4}\.\d\.\d{5}\.[A-Z]\b/i;

function asText(value) {
    return String(value ?? "").trim();
}

function nodeId(type, value) {
    return `${type}:${asText(value).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")}`;
}

function rowValue(row, names) {
    for (const name of names) {
        const value = asText(row?.[name]);
        if (value) return value;
    }
    return "";
}

function extractProjectCode(value) {
    const raw = asText(value);
    const match = raw.match(PROJECT_CODE_RE);
    return match ? match[0].toUpperCase() : raw;
}

function normalizeMatchText(value) {
    return asText(value).toLowerCase().replace(/\s+/g, " ");
}

function addNode(map, node) {
    if (!node.id) return;
    const existing = map.get(node.id);
    if (existing) {
        map.set(node.id, {
            ...existing,
            ...node,
            count: Math.max(existing.count || 0, node.count || 0),
        });
        return;
    }
    map.set(node.id, node);
}

function addEdge(map, edge) {
    if (!edge.from || !edge.to || edge.from === edge.to) return;
    const key = `${edge.from}->${edge.to}:${edge.label || ""}`;
    const existing = map.get(key);
    if (existing) {
        map.set(key, {
            ...existing,
            strength: Math.max(existing.strength || 1, edge.strength || 1),
        });
        return;
    }
    map.set(key, edge);
}

function sortedEntries(counter, limit) {
    return [...counter.entries()]
        .sort((a, b) => b[1].count - a[1].count || a[0].localeCompare(b[0]))
        .slice(0, limit);
}

function collectGraphInputs(messages) {
    const dataTables = [];
    const papers = [];

    for (const message of messages || []) {
        if (message?.dataTable && !["alma_products", "alma_project_picker"].includes(message.dataTable.tableKind || "")) {
            dataTables.push(message.dataTable);
        }
        if (Array.isArray(message?.papers)) papers.push(...message.papers);
    }

    return { dataTables, papers };
}

function normalizeObservationLinks(paper) {
    const rawLinks = Array.isArray(paper?.observationLinks)
        ? paper.observationLinks
        : Array.isArray(paper?.observation_links)
            ? paper.observation_links
            : [];
    return rawLinks
        .map((link) => ({
            identifier: asText(link.identifier),
            identifierType: asText(link.identifierType || link.identifier_type || "identifier"),
            relation: asText(link.relation || "explicit identifier"),
            confidence: asText(link.confidence || "explicit"),
            adsQuery: asText(link.adsQuery || link.ads_query),
        }))
        .filter((link) => link.identifier);
}

export function buildObservationPaperGraph(messages, options = {}) {
    const { dataTables, papers } = collectGraphInputs(messages);
    const projectLimit = options.projectLimit ?? 7;
    const targetLimit = options.targetLimit ?? 7;
    const paperLimit = options.paperLimit ?? papers.length;

    const papersWithIdentifierLinks = papers.filter((paper) => normalizeObservationLinks(paper).length > 0);
    if (dataTables.length === 0 && papersWithIdentifierLinks.length === 0) return null;

    const nodes = new Map();
    const edges = new Map();
    const projects = new Map();
    const targets = new Map();
    const archives = new Map();
    const explicitIdentifiers = new Map();

    let observationCount = 0;

    for (const table of dataTables) {
        const archiveLabel = asText(table.sourceName) || "Archive";
        const archiveId = nodeId("archive", archiveLabel);
        addNode(nodes, {
            id: archiveId,
            type: "archive",
            label: archiveLabel,
            detail: "Observation archive",
            count: table.rows?.length || 0,
            url: table.archiveLink || "",
        });
        archives.set(archiveId, archiveLabel);

        for (const row of table.rows || []) {
            observationCount += 1;
            const project = extractProjectCode(
                rowValue(row, ["Proposal ID", "Project", "project_code", "proposal_id", "obs_publisher_did"])
            ) || "Unlabeled project";
            const target = rowValue(row, ["Target", "target_name", "Source", "Object"]) || "Unlabeled target";
            const telescope = rowValue(row, ["Telescope", "Instrument", "obs_collection", "instrument_name"]);
            const band = rowValue(row, ["Band", "band_list"]);
            const projectId = nodeId("project", project);
            const targetId = nodeId("target", target);

            const projectEntry = projects.get(projectId) || {
                label: project,
                count: 0,
                archiveId,
                detailParts: new Set(),
                url: row._link || table.archiveLink || "",
            };
            projectEntry.count += 1;
            if (telescope) projectEntry.detailParts.add(telescope);
            if (band) projectEntry.detailParts.add(band);
            if (!projectEntry.url && row._link) projectEntry.url = row._link;
            projects.set(projectId, projectEntry);

            const targetEntry = targets.get(targetId) || {
                label: target,
                count: 0,
                detailParts: new Set(),
            };
            targetEntry.count += 1;
            if (project && project !== "Unlabeled project") targetEntry.detailParts.add(project);
            targets.set(targetId, targetEntry);

            addEdge(edges, {
                from: archiveId,
                to: projectId,
                label: "contains",
                strength: projectEntry.count,
            });
            addEdge(edges, {
                from: projectId,
                to: targetId,
                label: "observes",
                strength: targetEntry.count,
            });
        }
    }

    const topProjects = sortedEntries(projects, projectLimit);
    const topTargets = sortedEntries(targets, targetLimit);

    for (const [projectId, project] of topProjects) {
        addNode(nodes, {
            id: projectId,
            type: "project",
            label: project.label,
            detail: [...project.detailParts].slice(0, 3).join(" / ") || "Observation program",
            count: project.count,
            url: project.url,
        });
    }

    for (const [targetId, target] of topTargets) {
        addNode(nodes, {
            id: targetId,
            type: "target",
            label: target.label,
            detail: `${target.count} observation${target.count === 1 ? "" : "s"}`,
            count: target.count,
        });
    }

    const visibleProjectIds = new Set(topProjects.map(([id]) => id));
    const visibleTargetIds = new Set(topTargets.map(([id]) => id));
    for (const [key, edge] of [...edges.entries()]) {
        if (
            (edge.from.startsWith("project:") && !visibleProjectIds.has(edge.from)) ||
            (edge.to.startsWith("project:") && !visibleProjectIds.has(edge.to)) ||
            (edge.from.startsWith("target:") && !visibleTargetIds.has(edge.from)) ||
            (edge.to.startsWith("target:") && !visibleTargetIds.has(edge.to))
        ) {
            edges.delete(key);
        }
    }

    const targetEntries = topTargets.map(([id, target]) => ({
        id,
        label: target.label,
        normalized: normalizeMatchText(target.label),
    })).filter((target) => target.normalized.length >= 3);

    const projectEntries = topProjects.map(([id, project]) => ({
        id,
        label: project.label,
        normalized: normalizeMatchText(project.label),
    })).filter((project) => PROJECT_CODE_RE.test(project.label));

    const rankedPapers = [...papers]
        .sort((a, b) => Number(b.citationCount || 0) - Number(a.citationCount || 0))
        .slice(0, paperLimit);

    let linkedPapers = 0;
    for (const paper of rankedPapers) {
        const paperKey = paper.bibcode || paper.doi || paper.id || paper.title;
        const paperId = nodeId("paper", paperKey);
        const paperText = normalizeMatchText(`${paper.title || ""} ${paper.abstract || ""}`);
        const paperUrl = paper.bibcode
            ? `https://ui.adsabs.harvard.edu/abs/${encodeURIComponent(paper.bibcode)}`
            : paper.doi
                ? `https://doi.org/${paper.doi}`
                : paper.arxivId
                    ? `https://arxiv.org/abs/${paper.arxivId}`
                    : "";
        addNode(nodes, {
            id: paperId,
            type: "paper",
            label: paper.title || "Untitled paper",
            detail: `${paper.year || "n.d."} / ${Number(paper.citationCount || 0)} citations`,
            count: Number(paper.citationCount || 0),
            url: paperUrl,
        });

        let directLinks = 0;
        const observationLinks = normalizeObservationLinks(paper);
        for (const link of observationLinks) {
            const identifierId = nodeId("identifier", link.identifier);
            const existing = explicitIdentifiers.get(identifierId) || {
                label: link.identifier,
                count: 0,
                identifierType: link.identifierType,
                confidence: link.confidence,
            };
            existing.count += 1;
            explicitIdentifiers.set(identifierId, existing);
            addNode(nodes, {
                id: identifierId,
                type: "identifier",
                label: link.identifier,
                detail: link.identifierType.replace(/_/g, " ") || "archive identifier",
                count: existing.count,
            });
            addEdge(edges, {
                from: identifierId,
                to: paperId,
                label: link.relation || "explicit id",
                strength: link.confidence === "explicit" ? 4 : 2,
            });
            directLinks += 1;
        }
        for (const project of projectEntries) {
            if (paperText.includes(project.normalized.toLowerCase())) {
                addEdge(edges, {
                    from: project.id,
                    to: paperId,
                    label: "mentions project",
                    strength: 3,
                });
                directLinks += 1;
            }
        }
        for (const target of targetEntries) {
            if (paperText.includes(target.normalized)) {
                addEdge(edges, {
                    from: target.id,
                    to: paperId,
                    label: "mentions target",
                    strength: 2,
                });
                directLinks += 1;
            }
        }

        if (directLinks === 0 && dataTables.length > 0) {
            const contextId = "provenance:session-literature";
            addNode(nodes, {
                id: contextId,
                type: "provenance",
                label: "Session literature",
                detail: "Returned with this research context",
                count: rankedPapers.length,
            });
            addEdge(edges, {
                from: contextId,
                to: paperId,
                label: "context",
                strength: 1,
            });
        } else {
            linkedPapers += 1;
        }
    }

    const graphNodes = [...nodes.values()];
    const graphEdges = [...edges.values()].filter((edge) =>
        nodes.has(edge.from) && nodes.has(edge.to)
    );

    if (graphNodes.length < 2) return null;

    return {
        nodes: graphNodes,
        edges: graphEdges,
        summary: {
            archives: archives.size,
            projects: projects.size,
            identifiers: explicitIdentifiers.size,
            observations: observationCount,
            papers: papers.length,
            linkedPapers,
        },
    };
}
