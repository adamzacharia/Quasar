/**
 * Serialized async render chain (guard CX-07, task-3c99b37-115 round 2).
 *
 * PlotlyCard renders into ONE shared DOM element. React's effect cleanup
 * can set a `cancelled` flag and purge the element, but it cannot cancel an
 * already-running `Plotly.newPlot` — so without ordering, an OLD theme
 * render finishing late could overwrite a NEWER one. `createRenderChain`
 * restores a total order:
 *
 * - tasks run strictly one at a time, in enqueue order: each waits for the
 *   previous task to settle before starting, so a late-finishing old render
 *   can never land AFTER a newer render has started painting;
 * - enqueueing a task marks every earlier task STALE; the running task is
 *   handed an `isStale()` probe so it can skip its follow-up mutations
 *   after every await point, and a stale task that has not started yet is
 *   skipped entirely;
 * - a task that throws rejects ITS returned promise but never wedges the
 *   chain for the tasks behind it.
 *
 * Kept as plain JS beside plotly-theme.js so `node --test` can assert the
 * ordering invariant without a DOM/React harness.
 */

/** @typedef {(isStale: () => boolean) => (void | Promise<void>)} RenderTask */

export function createRenderChain() {
    /** @type {Promise<void>} */
    let chain = Promise.resolve();
    let generation = 0;
    return {
        /**
         * Enqueue `task` behind every previously enqueued task.
         * @param {RenderTask} task
         * @returns {Promise<void>} settles when THIS task (or its skip) is done
         */
        enqueue(task) {
            const gen = ++generation;
            const isStale = () => gen !== generation;
            const run = chain.then(() => {
                if (isStale()) return undefined;
                return task(isStale);
            });
            // Failures surface to the enqueuer via `run`, but the chain
            // itself always continues for later renders.
            chain = run.then(() => undefined, () => undefined);
            return run;
        },
    };
}
