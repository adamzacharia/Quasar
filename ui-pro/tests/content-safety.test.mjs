import test from "node:test";
import assert from "node:assert/strict";

import {
    isBlockedWebUrl,
    isSafeWebImage,
    isSafeWebSource,
    looksExplicitWebText,
    safeAssistantWebText,
} from "../src/lib/content-safety.js";

test("blocks known adult domains without substring false positives", () => {
    assert.equal(isBlockedWebUrl("https://www.xvideos.com/watch/123"), true);
    assert.equal(isBlockedWebUrl("https://xvideos.com.example.org/article"), false);
});

test("blocks explicit source metadata", () => {
    assert.equal(
        isSafeWebSource({
            url: "https://example.com/profile",
            title: "Biography",
            snippet: "Pornographic actress and director.",
        }),
        false,
    );
    assert.equal(
        isSafeWebSource({
            url: "https://example.edu/classes",
            title: "Adult education",
            snippet: "Continuing education.",
        }),
        true,
    );
    assert.equal(looksExplicitWebText("ordinary astronomy article"), false);
});

test("general web images are rejected by the UI defense", () => {
    assert.equal(
        isSafeWebImage({ url: "https://example.com/image.jpg", description: "Portrait" }),
        false,
    );
});

test("stored assistant text with explicit web content is replaced", () => {
    assert.equal(
        safeAssistantWebText("The result describes a pornographic actress."),
        "Some web results were withheld by the safety filter.",
    );
    assert.equal(safeAssistantWebText("ALMA Band 6 calibration guide."), "ALMA Band 6 calibration guide.");
});
