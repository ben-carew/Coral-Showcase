"""Plot styling: one palette, and identity that follows the arm.

Two rules do most of the work here.

**Colour follows the entity, never its position.** An arm keeps the same colour
in every figure of a report, and adding or removing an arm does not repaint the
others. Matplotlib's default behaviour is the opposite -- it cycles colours by
plotting order, so dropping one arm silently recolours the rest and two figures
of the same report disagree about which line is Coral.

**Identity is never carried by colour alone.** Each arm gets a linestyle and a
marker alongside its colour, so the figures survive greyscale printing and
colour-vision deficiency. The palette is the validated eight-slot categorical
set (worst adjacent CVD Delta E 9.1, normal-vision 19.6); three of its light-mode
slots fall below 3:1 contrast on white, which is why every figure in a Sextant
report is accompanied by the same numbers as a table.
"""

from __future__ import annotations

import math
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt

#: Validated categorical palette, in fixed assignment order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)),
              (0, (1, 1)), (0, (7, 2, 1, 2))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

#: Beyond this many arms a categorical palette stops being readable; the report
#: says so rather than inventing a ninth hue.
MAX_DISTINGUISHABLE_ARMS = len(PALETTE)

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOUR = "#d8d7d2"
TRUTH_COLOUR = "#52514e"

RC_PARAMS = {
    # Restated AFTER the science sheet, which sets several of these itself:
    # it disables the grid, puts ticks on all four sides and picks its own
    # figure size. A report of distributions is much easier to read off with a
    # light grid, and the page size is load-bearing (see PAGE_SIZE).
    "figure.figsize": (9.6, 9.6),
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.labelcolor": TEXT_PRIMARY,
    "axes.edgecolor": GRID_COLOUR,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID_COLOUR,
    "grid.alpha": 0.6,
    "grid.linewidth": 0.6,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "text.color": TEXT_PRIMARY,
}


#: SciencePlots' `science` sheet, for a report that has to sit beside published
#: HK figures. Applied FIRST, then RC_PARAMS on top: the sheet supplies the
#: typographic conventions (serif-ish sizing, inward ticks, thin frame) and this
#: module keeps the things a Sextant report cannot delegate -- the categorical
#: palette, and the rule that identity never rides on colour alone.
#:
#: `no-latex` is not optional here: the plain `science` style routes text through
#: a LaTeX installation, and there is none on the cluster or in the HyperK conda
#: environment, so every figure would fail to render.
SCIENCE_STYLE = ["science", "no-latex"]


def use_style():
    """Apply the SciencePlots base, then Sextant's own overrides."""
    try:
        import scienceplots            # noqa: F401  (registers the style sheets)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.style.use(SCIENCE_STYLE)
    except Exception as exc:           # missing package, or a broken sheet
        warnings.warn(f"SciencePlots unavailable ({exc}); "
                      f"falling back to Sextant's own rcParams only")
    mpl.rcParams.update(RC_PARAMS)


class ArmStyle:
    """Stable colour / linestyle / marker per arm label.

    Built once per report and passed to every plotting function, so identity is
    consistent across all figures. Labels are assigned in the order first seen.
    """

    def __init__(self, labels=(), reference=None):
        self._assigned = {}
        self.reference = reference
        for label in labels:
            self.add(label)

    def add(self, label: str) -> dict:
        if label not in self._assigned:
            index = len(self._assigned)
            self._assigned[label] = {
                "color": PALETTE[index % len(PALETTE)],
                "linestyle": LINESTYLES[index % len(LINESTYLES)],
                "marker": MARKERS[index % len(MARKERS)],
                "index": index,
            }
        return self._assigned[label]

    def __getitem__(self, label: str) -> dict:
        return self.add(label)

    def color(self, label): return self[label]["color"]
    def linestyle(self, label): return self[label]["linestyle"]
    def marker(self, label): return self[label]["marker"]

    # The arm is named `arm`, not `label`, so that callers can override
    # matplotlib's own `label=` (to append an AUC, say) without colliding with
    # the positional argument.
    def line_kwargs(self, arm, **overrides) -> dict:
        style = self[arm]
        kwargs = {"color": style["color"], "linestyle": style["linestyle"],
                  "label": arm}
        kwargs.update(overrides)
        return kwargs

    def point_kwargs(self, arm, **overrides) -> dict:
        style = self[arm]
        kwargs = {"color": style["color"], "marker": style["marker"],
                  "linestyle": style["linestyle"], "label": arm}
        kwargs.update(overrides)
        return kwargs

    @property
    def labels(self) -> list:
        return list(self._assigned)

    def overflow(self) -> bool:
        return len(self._assigned) > MAX_DISTINGUISHABLE_ARMS


def finish(ax, *, title=None, xlabel=None, ylabel=None, legend=None,
           legend_loc="best"):
    """Apply the shared axis furniture.

    `legend` defaults to "whenever there is more than one labelled series" --
    a single series is named by the title and a one-entry legend is noise.
    """
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    handles, labels = ax.get_legend_handles_labels()
    show = (len(labels) > 1) if legend is None else legend
    if show and labels:
        ax.legend(loc=legend_loc)
    return ax


#: Every page of a report is this size. A PDF whose pages change shape from one
#: figure to the next cannot be printed, bound or flipped through, so the panel
#: grid adapts to the page rather than the page to the grid.
#:
#: SQUARE, deliberately. Every panel in a Sextant report is a distribution or a
#: residual profile, and a square data area is the honest default for those: a
#: wide box exaggerates horizontal separation between arms, which is exactly the
#: judgement the reader is being asked to make. A square page is what lets a
#: square panel grid fill it without leaving a band of white down one side.
PAGE_SIZE = (9.6, 9.6)             # inches, square


def square_axes(ax):
    """Force a 1:1 DATA BOX, independent of the units on either axis.

    `set_box_aspect` constrains the drawn box, unlike `set_aspect`, which ties
    the two data scales together and would be meaningless for, say, centimetres
    against degrees.
    """
    try:
        ax.set_box_aspect(1)
    except Exception:                      # very old matplotlib; not fatal
        pass
    return ax


def grid_figure(n_panels, ncols=None, panel_size=None, **kwargs):
    """A page-sized figure with n_panels SQUARE axes on a near-square grid.

    `panel_size` is accepted and ignored: it survives only so existing callers
    keep working. Panels are sized by dividing the page, not the other way
    round -- see PAGE_SIZE.

    `ncols` is ADVISORY and treated as an upper bound. The layout wants
    ceil(sqrt(n)) columns so the grid is as square as the panel count allows;
    several callers ask for a flat 1xN strip, which was right on a landscape
    page and leaves most of a square one empty. Passing a SMALLER ncols still
    works, so a caller that deliberately wants a narrow figure keeps it.
    """
    n_panels = max(int(n_panels), 1)
    square_cols = int(math.ceil(math.sqrt(n_panels)))
    ncols = square_cols if ncols is None else min(int(ncols), square_cols)
    ncols = max(1, min(ncols, n_panels))
    nrows = -(-n_panels // ncols)
    fig = plt.figure(figsize=PAGE_SIZE, **kwargs)
    # A single panel spanning the whole gridspec fills the sheet edge to edge,
    # which reads as a blown-up screenshot rather than a figure. Give a lone
    # panel real margins; a full grid still uses the page.
    margins = ({"left": 0.16, "right": 0.84, "bottom": 0.20, "top": 0.80}
               if n_panels == 1 else {})
    gs = fig.add_gridspec(nrows, 2 * ncols, **margins)  # half-columns, so a short last
    used, axs = [], []                           # row can be centred on the page
    for row in range(nrows):
        in_row = min(ncols, n_panels - row * ncols)
        if in_row <= 0:
            break
        offset = (ncols - in_row)                # half-column shift, centres it
        for col in range(in_row):
            c0 = offset + 2 * col
            axs.append(fig.add_subplot(gs[row, c0:c0 + 2]))
    for ax in axs:
        square_axes(ax)
    return fig, axs


def annotate_count(ax, n, *, loc="upper right"):
    """Put the sample size on the plot -- a distribution without n is unreadable."""
    x, ha = (0.98, "right") if "right" in loc else (0.02, "left")
    y, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(x, y, f"n = {n:,}", transform=ax.transAxes, ha=ha, va=va,
            fontsize=8, color=TEXT_SECONDARY)


def caption(fig, text, *, fontsize: int = 11, width: int = 118):
    """Put a wrapped caption above a figure, inside the sheet.

    The report writes PDF pages with `bbox_inches=None` so every page is the
    same shape, which means anything drawn above y=1.0 is simply off the paper
    and is lost -- the standalone PNGs, cropped tight, kept it and hid the
    problem. So the caption sits just inside the top edge and the layout is
    given matching headroom, rather than being placed at y=1.01 and trusting
    the crop.
    """
    text = wrap_caption(str(text), width)
    lines = text.count("\n") + 1
    head = min(0.06 + 0.028 * lines, 0.22)     # room for the caption itself
    fig.tight_layout(rect=(0, 0, 1, 1 - head))
    fig.suptitle(text, y=1 - 0.35 * head, fontsize=fontsize,
                 va="center")
    return fig


def wrap_caption(text: str, width: int = 118) -> str:
    """Wrap a figure caption so it cannot run off the page.

    Figure captions are drawn with `fig.suptitle`, which does not wrap: a long
    one simply extends past the sheet and the ends are lost. Explicit newlines
    in the caption are preserved.
    """
    import textwrap

    return "\n".join(
        line for para in str(text).split("\n")
        for line in (textwrap.wrap(para, width) or [""]))
