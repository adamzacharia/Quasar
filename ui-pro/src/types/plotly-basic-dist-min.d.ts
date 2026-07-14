// Minimal module declaration for the Plotly "basic" partial bundle.
// No upstream @types package covers the *-dist-min entry points, so we
// declare only the surface PlotlyCard actually uses.
declare module "plotly.js-basic-dist-min" {
    export interface PlotlyModule {
        newPlot(
            root: HTMLElement,
            data: unknown[],
            layout?: Record<string, unknown>,
            config?: Record<string, unknown>,
        ): Promise<unknown>;
        purge(root: HTMLElement): void;
        restyle(
            root: HTMLElement,
            update: Record<string, unknown>,
            traceIndices?: number[],
        ): Promise<unknown>;
        toImage(
            root: HTMLElement,
            options?: {
                format?: "png" | "svg" | "jpeg" | "webp";
                width?: number;
                height?: number;
                scale?: number;
            },
        ): Promise<string>;
    }
    const Plotly: PlotlyModule;
    export default Plotly;
}
