from research.plot.engine import MultiPanelEngine
import pandas as pd
import os

def plotStabilityHeatmaps(margin: int = 20):
    """
    Generates 2D heatmaps comparing the 'Rescued' and 'Naturally Stable' 
    sets for a given margin using the themed PlotEngine.
    """
    results_dir = "results"
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    rescuedFile = os.path.join(results_dir, f"stability_margin_{margin}_rescued.parquet")
    stableFile = os.path.join(results_dir, f"stability_margin_{margin}_stable.parquet")

    sets = [
        ("Rescued (Wht Hero)", rescuedFile, "magma"),
        ("Naturally Stable", stableFile, "viridis")
    ]

    # Use MultiPanelEngine for a professional themed layout
    multi = MultiPanelEngine(nrows=1, ncols=2, figsize=(16, 7))

    for idx, (title, path, cmap) in enumerate(sets):
        if not os.path.exists(path):
            print(f"Skipping {title}: {path} not found.")
            continue

        df = pd.read_parquet(path)
        
        pivot_data = df.pivot_table(
            index='K', 
            columns='order', 
            values='wht_logCond', 
            aggfunc='mean'
        )

        # Get themed PlotEngine for this panel
        engine = multi.getEngine(idx)
        engine.addHeatmap(
            pivot_data, 
            xlabel="Polynomial Order (N)", 
            ylabel="Lobe Count (K)",
            cmap=cmap, 
            cbarLabel="Log10(Condition Number)"
        )
        engine.setTitle(f"{title} Landscape (Margin {margin}nm)")

    outputPath = os.path.join(plots_dir, f"stability_heatmap_margin_{margin}.png")
    multi.saveFigure(outputPath, dpi=300)
    print(f"Heatmap saved to: {outputPath}")

if __name__ == "__main__":
    # Generate heatmaps for all margins to compare the ringing fix
    for m in [0, 10, 20]:
        plotStabilityHeatmaps(m)
    print("\nVisualizations complete.")
