import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional, Callable, Dict, List, Tuple, Any

class PlotEngine:
    """
    Core plotting engine for single-panel scientific figures.
    """

    sColors = {
        'primary': '#5BC0DE',
        'secondary': '#F0E68C',
        'tertiary': '#90C695',
        'figureBg': '#0f1116',
        'axesBg': '#0f1116',
        'grid': '#FFFFFF',
        'spine': '#FFFFFF',
        'text': '#FFFFFF',
        'tick': '#FFFFFF',
    }

    sFontSizes = {
        'suptitle': 14,
        'title': 11,
        'label': 10,
        'tick': 8,
        'legend': 8,
        'annotation': 8,
    }

    def __init__(self, figsize: Tuple[float, float] = (10, 6)):
        self.m_figure = None
        self.m_axes = None
        self.m_figsize = figsize
        self.m_lineCounter = 0

        self.internalInitializeFigure()
        self.internalApplyTheme()

    def internalInitializeFigure(self) -> None:
        self.m_figure, self.m_axes = plt.subplots(figsize=self.m_figsize, facecolor=self.sColors['figureBg'])
        self.m_figure.patch.set_facecolor(self.sColors['figureBg'])
        self.m_axes.set_facecolor(self.sColors['axesBg'])
        self.m_axes.patch.set_facecolor(self.sColors['axesBg'])

    def internalApplyTheme(self) -> None:
        for spine in self.m_axes.spines.values():
            spine.set_edgecolor(self.sColors['spine'])
            spine.set_linewidth(0.8)

        self.m_axes.grid(True, which='major', linestyle='-',
                         linewidth=0.5, alpha=0.12, color=self.sColors['grid'])

        self.m_axes.tick_params(
            axis='both',
            which='major',
            labelsize=self.sFontSizes['tick'],
            colors=self.sColors['tick'],
            length=4,
            width=0.8,
            direction='out'
        )

        self.m_axes.xaxis.label.set_color(self.sColors['text'])
        self.m_axes.yaxis.label.set_color(self.sColors['text'])

    def addLine(self, xData: np.ndarray, yData: np.ndarray,
                label: Optional[str] = None,
                color: Optional[str] = None,
                linewidth: float = 1.5,
                linestyle: str = '-',
                alpha: float = 1.0) -> None:
        if color is None:
            colorCycle = [self.sColors['primary'],
                          self.sColors['secondary'],
                          self.sColors['tertiary']]
            color = colorCycle[self.m_lineCounter % len(colorCycle)]
            self.m_lineCounter += 1

        self.m_axes.plot(xData, yData, label=label, color=color,
                         linewidth=linewidth, linestyle=linestyle, alpha=alpha)
        self.m_axes.margins(x=0.01)

    def addScatter(self, xData: np.ndarray, yData: np.ndarray,
                   label: Optional[str] = None,
                   color: Optional[str] = None,
                   marker: str = 'o',
                   size: float = 30,
                   alpha: float = 0.7) -> None:
        if color is None:
            color = self.sColors['primary']
        self.m_axes.scatter(xData, yData, label=label, color=color,
                            marker=marker, s=size, alpha=alpha, edgecolors='none')

    def setTitle(self, title: str) -> None:
        self.m_axes.set_title(title, fontsize=self.sFontSizes['title'],
                               color=self.sColors['text'], pad=10)

    def setLabels(self, xlabel: str, ylabel: str) -> None:
        self.m_axes.set_xlabel(xlabel, fontsize=self.sFontSizes['label'])
        self.m_axes.set_ylabel(ylabel, fontsize=self.sFontSizes['label'])

    def addLegend(self, location: str = 'best',
                  framealpha: float = 0.5,
                  edgecolor: Optional[str] = None) -> None:
        if edgecolor is None:
            edgecolor = self.sColors['spine']

        legend = self.m_axes.legend(
            loc=location,
            fontsize=self.sFontSizes['legend'],
            framealpha=framealpha,
            edgecolor=edgecolor,
            fancybox=False,
            shadow=False,
            borderpad=0.3
        )
        legend.get_frame().set_facecolor(self.sColors['axesBg'])
        for text in legend.get_texts():
            text.set_color(self.sColors['text'])

    def saveFigure(self, filepath: str, dpi: int = 300,
                   transparentBackground: bool = False) -> None:
        self.m_figure.savefig(filepath, dpi=dpi,
                              facecolor=self.m_figure.get_facecolor(),
                              transparent=transparentBackground,
                              bbox_inches='tight')

    def show(self) -> None:
        plt.show()

    def clear(self) -> None:
        self.m_axes.clear()
        self.internalApplyTheme()
        self.m_lineCounter = 0


class MultiPanelEngine:
    """
    Multi-panel plotting engine for vertically stacked subplots.
    """

    def __init__(self, nrows: int, ncols: int = 1,
                 figsize: Tuple[float, float] = (10, 12),
                 sharex: bool = False,
                 compact: bool = False,
                 projection=None):
        self.m_nrows = nrows
        self.m_ncols = ncols
        self.m_figsize = figsize
        self.m_sharex = sharex
        self.m_figure = None
        self.m_panels = []
        self.m_compact = compact
        self.m_projection = projection
        self.internalInitializePanels()

    def internalInitializePanels(self) -> None:
        self.m_figure, axesArray = plt.subplots(
            self.m_nrows,
            self.m_ncols,
            figsize=self.m_figsize,
            sharex=self.m_sharex,
            facecolor=PlotEngine.sColors['figureBg'],
            constrained_layout=True,
            subplot_kw={'projection': self.m_projection} if self.m_projection else None
        )

        if self.m_nrows == 1 and self.m_ncols == 1:
            axesList = [axesArray]
        elif self.m_nrows == 1 or self.m_ncols == 1:
            axesList = axesArray if isinstance(axesArray, np.ndarray) else [axesArray]
        else:
            axesList = np.array(axesArray).reshape(-1)

        self.m_figure.patch.set_facecolor(PlotEngine.sColors['figureBg'])

        for ax in axesList:
            ax.set_facecolor(PlotEngine.sColors['axesBg'])
            ax.patch.set_facecolor(PlotEngine.sColors['axesBg'])
            panel = self.internalCreatePanelFromAxes(ax)
            self.m_panels.append(panel)

    def internalCreatePanelFromAxes(self, ax) -> PlotEngine:
        panel = PlotEngine.__new__(PlotEngine)
        panel.m_figure = self.m_figure
        panel.m_axes = ax
        panel.m_figsize = self.m_figsize
        panel.m_lineCounter = 0
        panel.internalApplyTheme()
        return panel

    def getPanel(self, index: int) -> PlotEngine:
        return self.m_panels[index]

    def setMainTitle(self, title: str, fontSize: Optional[int] = None) -> None:
        if fontSize is None:
            fontSize = PlotEngine.sFontSizes['suptitle']
        self.m_figure.suptitle(title, fontsize=fontSize, color=PlotEngine.sColors['text'])

    def saveFigure(self, filepath: str, dpi: int = 300) -> None:
        self.m_figure.savefig(filepath, dpi=dpi, facecolor=self.m_figure.get_facecolor(), bbox_inches='tight')

    def show(self) -> None:
        plt.show()


class AnimationEngine:
    """
    Animation engine for deterministic scientific animations.
    """

    def __init__(self, plotEngine: PlotEngine):
        self.m_plotEngine = plotEngine
        self.m_animation = None

    def animate(self, updateFn: Callable[[int], List], frames: int, interval: int = 50, useBlit: bool = True) -> None:
        self.m_animation = animation.FuncAnimation(
            self.m_plotEngine.m_figure,
            updateFn,
            frames=frames,
            interval=interval,
            blit=useBlit,
            repeat=True
        )

    def saveAnimation(self, filepath: str, fps: int = 30, codec: str = 'h264') -> None:
        if filepath.endswith('.gif'):
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps, codec=codec, bitrate=1800)
        self.m_animation.save(filepath, writer=writer, dpi=100)

    def show(self) -> None:
        plt.show()


class SurfaceEngine(PlotEngine):
    """
    3D surface plotting engine.
    """
    def internalInitializeFigure(self) -> None:
        self.m_figure = plt.figure(figsize=self.m_figsize, facecolor=self.sColors['figureBg'])
        self.m_axes = self.m_figure.add_subplot(111, projection='3d')
        self.m_axes.set_facecolor(self.sColors['axesBg'])

    def addSurface(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, cmap: str = 'viridis', alpha: float = 0.95):
        surf = self.m_axes.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=alpha)
        self.m_figure.colorbar(surf, shrink=0.6, aspect=12)

    def setView(self, elev: float = 30, azim: float = 45):
        self.m_axes.view_init(elev=elev, azim=azim)
