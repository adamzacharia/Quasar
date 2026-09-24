# IMG-02 imaging-word false positive (expect_web: false)

Question: "Explain how the ALMA imaging pipeline chooses the cell size and image size for continuum imaging."

T (40): a knowledge / documentation question. The word "imaging" must NOT route to an image search and no web search runs (documentation RAG is fine).

Key facts (C, 50): cell size set from the synthesized beam (a fraction of the beam, about 5 pixels across the minor axis or similar heuristics) (20); image size from the primary beam / field of view (a multiple of the primary beam FWHM, e.g. covering the 20 percent level, mosaic extent) (20); mentions the pipeline heuristics / imaging parameters are in the pipeline weblog or documentation (10).

P (10): no image tiles, no web clutter.
