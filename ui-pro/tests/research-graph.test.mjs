import test from "node:test";
import assert from "node:assert/strict";

import { buildObservationPaperGraph } from "../src/lib/research-graph.js";

function dataMessage() {
    return {
        id: "data-1",
        role: "assistant",
        type: "data",
        content: "",
        dataTable: {
            sourceName: "ALMA",
            archiveLink: "https://almascience.nrao.edu/aq/",
            columns: ["Proposal ID", "Target", "Band", "Telescope"],
            metrics: [],
            rows: [
                {
                    "Proposal ID": "2019.1.00123.S",
                    Target: "M87",
                    Band: "Band 6",
                    Telescope: "ALMA",
                    _link: "https://almascience.nrao.edu/aq/?project_code=2019.1.00123.S",
                },
                {
                    "Proposal ID": "2019.1.00123.S",
                    Target: "M87",
                    Band: "Band 7",
                    Telescope: "ALMA",
                },
            ],
        },
    };
}

test("builds archive, project, target, and paper nodes from chat messages", () => {
    const graph = buildObservationPaperGraph([
        dataMessage(),
        {
            id: "papers-1",
            role: "assistant",
            type: "papers",
            content: "",
            papers: [
                {
                    id: "paper-1",
                    title: "ALMA observations of M87 in project 2019.1.00123.S",
                    authors: "A. Astronomer",
                    year: 2024,
                    journal: "ApJ",
                    citationCount: 22,
                    type: "journal",
                    bibcode: "2024ApJ...000...1A",
                    abstract: "We analyze M87 with ALMA.",
                },
            ],
        },
    ]);

    assert.ok(graph);
    assert.equal(graph.summary.observations, 2);
    assert.equal(graph.summary.projects, 1);
    assert.equal(graph.summary.papers, 1);
    assert.equal(graph.summary.linkedPapers, 1);
    assert.ok(graph.nodes.some((node) => node.type === "archive" && node.label === "ALMA"));
    assert.ok(graph.nodes.some((node) => node.type === "project" && node.label === "2019.1.00123.S"));
    assert.ok(graph.nodes.some((node) => node.type === "target" && node.label === "M87"));
    assert.ok(graph.nodes.some((node) => node.type === "paper"));
    assert.ok(graph.edges.some((edge) => edge.label === "mentions target"));
    assert.ok(graph.edges.some((edge) => edge.label === "mentions project"));
});

test("does not show a graph for papers without observation context", () => {
    const graph = buildObservationPaperGraph([
        {
            id: "papers-only",
            role: "assistant",
            type: "papers",
            content: "",
            papers: [
                {
                    id: "paper-1",
                    title: "A paper without archive context",
                    authors: "A. Astronomer",
                    year: 2024,
                    journal: "ApJ",
                    citationCount: 5,
                    type: "journal",
                },
            ],
        },
    ]);

    assert.equal(graph, null);
});

test("does not treat ALMA product tables as observation graph input", () => {
    const graph = buildObservationPaperGraph([
        {
            id: "products",
            role: "assistant",
            type: "data",
            content: "",
            dataTable: {
                tableKind: "alma_products",
                sourceName: "ALMA Data Products",
                columns: ["File", "Product", "Proposal ID", "Target"],
                metrics: [],
                rows: [
                    {
                        File: "science.pbcor.fits",
                        Product: "primary-beam-corrected FITS",
                        "Proposal ID": "2016.1.00484.L",
                        Target: "AS 209",
                    },
                ],
            },
        },
    ]);

    assert.equal(graph, null);
});

test("builds an identifier-first graph from project linked papers without target context", () => {
    const graph = buildObservationPaperGraph([
        {
            id: "papers-by-id",
            role: "assistant",
            type: "papers",
            content: "",
            papers: [
                {
                    id: "paper-id-1",
                    title: "A paper found by exact project code",
                    authors: "A. Astronomer",
                    year: 2025,
                    journal: "ApJ",
                    citationCount: 7,
                    type: "journal",
                    observationLinks: [
                        {
                            identifier: "2019.1.00123.S",
                            identifierType: "project_code",
                            relation: "explicit_identifier_search",
                            confidence: "explicit",
                        },
                    ],
                },
            ],
        },
    ]);

    assert.ok(graph);
    assert.equal(graph.summary.observations, 0);
    assert.equal(graph.summary.identifiers, 1);
    assert.equal(graph.summary.linkedPapers, 1);
    assert.ok(graph.nodes.some((node) => node.type === "identifier" && node.label === "2019.1.00123.S"));
    assert.ok(graph.edges.some((edge) => edge.label === "explicit_identifier_search"));
});

test("renders every identifier-linked paper in the graph by default", () => {
    const papers = Array.from({ length: 14 }, (_, index) => ({
        id: `paper-id-${index + 1}`,
        title: `Identifier linked paper ${index + 1}`,
        authors: "A. Astronomer",
        year: 2020 + index,
        journal: "ApJ",
        citationCount: 100 - index,
        type: "journal",
        observationLinks: [
            {
                identifier: "2016.1.00484.L",
                identifierType: "project_code",
                relation: "explicit_identifier_search",
                confidence: "explicit",
            },
        ],
    }));

    const graph = buildObservationPaperGraph([
        {
            id: "papers-by-id",
            role: "assistant",
            type: "papers",
            content: "",
            papers,
        },
    ]);

    assert.ok(graph);
    assert.equal(graph.summary.papers, 14);
    assert.equal(graph.summary.linkedPapers, 14);
    assert.equal(graph.nodes.filter((node) => node.type === "paper").length, 14);
    assert.equal(graph.edges.filter((edge) => edge.label === "explicit_identifier_search").length, 14);
});

test("unmatched papers are connected through explicit session provenance", () => {
    const graph = buildObservationPaperGraph([
        dataMessage(),
        {
            id: "papers-1",
            role: "assistant",
            type: "papers",
            content: "",
            papers: [
                {
                    id: "paper-2",
                    title: "Unrelated molecular cloud survey",
                    authors: "B. Astronomer",
                    year: 2022,
                    journal: "A&A",
                    citationCount: 12,
                    type: "journal",
                },
            ],
        },
    ]);

    assert.ok(graph);
    assert.equal(graph.summary.linkedPapers, 0);
    assert.ok(graph.nodes.some((node) => node.type === "provenance" && node.label === "Session literature"));
    assert.ok(graph.edges.some((edge) => edge.label === "context"));
});
