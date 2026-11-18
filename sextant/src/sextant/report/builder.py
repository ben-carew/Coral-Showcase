"""Assemble the figures and tables into one PDF, plus a directory of outputs.

The output of a Sextant analysis is a directory:

    <outdir>/
        report.pdf          every figure and table, in reading order
        figures/*.png       the same figures individually, for slides
        tables/*.csv        every number in the report, machine-readable
        manifest.json       what was compared, and what produced it

The PDF is built with matplotlib's own PdfPages rather than a document
generator, so the whole reporting path has no dependency beyond what is already
needed to make the plots.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ..plots import style as st
from . import tables as table_io

#: One page size for the whole document, shared with the plotting style so a
#: figure page and a table page are the same shape.
PAGE_SIZE = st.PAGE_SIZE


class Report:
    """Collects figures and tables, then writes the whole output directory."""

    def __init__(self, title: str, outdir, *, subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle
        self.outdir = Path(outdir)
        self.sections = []        # (kind, heading, payload)
        self.tables = {}
        self.meta = {}
        #: Absolute paths are shortened against this in the rendered tables
        #: (the CSVs keep them in full).
        self.workspace = None

    # -- collection ----------------------------------------------------------

    def add_figure(self, heading: str, figure, *, name: str = None,
                   note: str = ""):
        if figure is None:
            return
        self.sections.append(("figure", heading,
                              {"figure": figure,
                               "name": name or _slug(heading), "note": note}))

    def add_table(self, heading: str, table, *, name: str = None,
                  note: str = "", max_rows: int = 30, workspace=None,
                  compact: bool = False):
        if table is None or table.empty:
            return
        # "run" throughout the output, whatever the internals call it. Done once
        # here so the CSV and the rendered page cannot disagree; a reconstruction
        # of one sample by one configuration is a run, and "arm" only ever meant
        # that.
        if "arm" in getattr(table, "columns", []):
            table = table.rename(columns={"arm": "run"})
        key = name or _slug(heading)
        self.tables[key] = table
        self.sections.append(("table", heading,
                              {"table": table, "name": key, "note": note,
                               "max_rows": max_rows, "compact": compact,
                               "workspace": workspace or self.workspace}))

    def add_csv(self, name: str, table):
        """A table written to tables/<name>.csv only, not rendered: for per-event
        output far too long for a page."""
        if table is not None and not table.empty:
            self.tables[name] = table

    def add_text(self, heading: str, body: str):
        self.sections.append(("text", heading, {"body": body}))

    # -- output --------------------------------------------------------------

    def write(self) -> Path:
        self.outdir.mkdir(parents=True, exist_ok=True)
        figures_dir = self.outdir / "figures"
        figures_dir.mkdir(exist_ok=True)

        pdf_path = self.outdir / "report.pdf"
        # bbox_inches=None, deliberately: the shared style sets savefig.bbox to
        # "tight", which crops each page to its own content and gives a PDF
        # whose pages are all different shapes. Cropping is right for the
        # standalone PNGs (a slide wants no margin) and wrong for a document.
        # The rc override, not a savefig kwarg: Figure.savefig fills bbox_inches
        # from rcParams with setdefault, and PdfPages routes through it, so the
        # style's "tight" setting wins over an explicit argument and every page
        # comes out cropped to its own content -- a PDF of ten different shapes.
        with plt.rc_context({"savefig.bbox": None}), PdfPages(pdf_path) as pdf:
            pdf.savefig(self._title_page())
            plt.close("all")
            for kind, heading, payload in self.sections:
                if kind == "figure":
                    figure = payload["figure"]
                    figure.set_size_inches(*PAGE_SIZE)
                    # The standalone PNG keeps the tight crop: a slide wants no
                    # margin, a bound document wants a uniform page.
                    figure.savefig(figures_dir / f"{payload['name']}.png",
                                   bbox_inches="tight")
                    pdf.savefig(figure)
                    plt.close(figure)
                elif kind == "table":
                    for page in self._table_pages(heading, payload):
                        pdf.savefig(page)
                        plt.close(page)
                else:
                    page = self._text_page(heading, payload["body"])
                    pdf.savefig(page)
                    plt.close(page)

        table_io.write_all(self.tables, self.outdir / "tables")
        (self.outdir / "manifest.json").write_text(json.dumps({
            "title": self.title, "subtitle": self.subtitle,
            "generated": datetime.now().isoformat(timespec="seconds"),
            "sections": [{"kind": k, "heading": h} for k, h, _ in self.sections],
            "tables": sorted(self.tables),
            **self.meta,
        }, indent=2, default=str))
        return pdf_path

    # -- page rendering ------------------------------------------------------

    #: Characters per line of the cover title before it is wrapped. At 26 pt on
    #: an A4 page a line runs off the sheet somewhere past this, and a title
    #: that runs off the page is worse than one on three lines.
    TITLE_WRAP = 46

    def _title_page(self):
        import textwrap

        fig = plt.figure(figsize=PAGE_SIZE)
        lines = textwrap.wrap(self.title, self.TITLE_WRAP) or [self.title]
        # Shrink a little for a title that needs three or more lines, so the
        # block stays inside the page whatever it is called.
        fontsize = 26 if len(lines) <= 2 else 21
        fig.text(0.5, 0.62, "\n".join(lines), ha="center", va="center",
                 fontsize=fontsize, color=st.TEXT_PRIMARY, linespacing=1.4)
        if self.subtitle:
            sub = "\n".join(textwrap.wrap(self.subtitle, 76) or [self.subtitle])
            fig.text(0.5, 0.62 - 0.045*len(lines) - 0.02, sub, ha="center",
                     va="top", fontsize=13, color=st.TEXT_SECONDARY,
                     linespacing=1.4)
        fig.text(0.5, 0.40,
                 datetime.now().strftime("generated %Y-%m-%d %H:%M"),
                 ha="center", va="center", fontsize=10, color=st.TEXT_SECONDARY)
        return fig

    def _text_page(self, heading, body):
        fig = plt.figure(figsize=PAGE_SIZE)
        fig.text(0.06, 0.93, heading, fontsize=16, va="top",
                 color=st.TEXT_PRIMARY)
        fig.text(0.06, 0.86, body, fontsize=10, va="top", ha="left",
                 color=st.TEXT_PRIMARY, wrap=True, family="monospace")
        return fig

    #: Total character budget across all columns of a table page. Set from the
    #: page width and the smallest font we are willing to print, and used to
    #: share width between columns.
    TABLE_CHAR_BUDGET = 100
    #: Never render a table below this. A table nobody can read is not a table;
    #: if the rows will not fit at this size the page is SPLIT (see
    #: _table_pages) rather than the type being shrunk to fit. The validation
    #: table used to come out at 5 pt this way, half the size of every other
    #: table in the report.
    TABLE_MIN_FONT = 9.0
    TABLE_MAX_FONT = 11.0

    def _table_pages(self, heading, payload):
        """One or more pages for a table, split so the type stays readable.

        The font used to be whatever made the rows fit, which on the validation
        table meant 5 pt. Rows per page are now chosen so TABLE_MIN_FONT always
        fits, and a table that needs more rooms gets more pages.
        """
        raw = payload["table"]
        # Compact FIRST. This used to format the raw table, decide the split
        # from it, and hand slices of it to _table_page -- which then skipped
        # its own compaction because it had been given a table. So the one
        # table that most needed folding was the one that never got it, and it
        # went out at 17 columns over two pages.
        if payload.get("compact"):
            raw = table_io.compact_summary(raw)
        table = table_io.format_for_display(raw, max_rows=payload["max_rows"])
        _, _, row_lines, _, header_lines, _ = _layout(table, self.TABLE_CHAR_BUDGET)

        axes_height_pt = 0.84 * PAGE_SIZE[1] * 72.0
        max_lines = max(int(axes_height_pt / (1.5 * self.TABLE_MIN_FONT)), 4)

        chunks, start, used = [], 0, 0
        for i, lines in enumerate(row_lines):
            if used and used + lines + header_lines > max_lines:
                chunks.append((start, i))
                start, used = i, 0
            used += lines
        chunks.append((start, len(row_lines)))

        pages = []
        for n, (lo, hi) in enumerate(chunks):
            part = table.iloc[lo:hi]
            head = heading if len(chunks) == 1 else \
                f"{heading}  ({n + 1} of {len(chunks)})"
            pages.append(self._table_page(head, payload, table=part))
        return pages

    def _table_page(self, heading, payload, table=None):
        raw = payload["table"]
        if table is None and payload.get("compact"):
            raw = table_io.compact_summary(raw)
        if table is None:
            table = table_io.format_for_display(raw, max_rows=payload["max_rows"])
        table = table_io.shorten_paths(table, payload.get("workspace"))
        table, transposed = table_io.maybe_transpose(table)
        widths, wrapped, row_lines, headers, header_lines, char_budget = _layout(
            table, self.TABLE_CHAR_BUDGET)

        fig = plt.figure(figsize=PAGE_SIZE)
        fig.text(0.06, 0.95, heading, fontsize=15, va="top",
                 color=st.TEXT_PRIMARY)
        # The grey explanatory note under the heading is deliberately NOT
        # drawn. It is still carried on the payload and still written to the
        # manifest, so nothing that reads the report loses it; it just does not
        # go on the page. Every table in this report is captioned by its own
        # column headers, and a paragraph of prose above each one crowded the
        # page for readers who already know what they are looking at.
        top = 0.89

        ax = fig.add_axes([0.05, 0.05, 0.90, top - 0.05])
        ax.axis("off")
        # "center left", not "upper left": a short table anchored to the top
        # leaves two thirds of the sheet blank under it. Centring costs a long
        # table nothing, since it fills the axes anyway.
        rendered = ax.table(cellText=wrapped, colLabels=headers,
                            colWidths=widths, loc="center left",
                            cellLoc="left", colLoc="left")
        rendered.auto_set_font_size(False)

        # Height is shared out by line count, so a three-line cell gets three
        # lines of room. Font size then comes from how many lines must fit --
        # this is what stops rows overlapping when a cell wraps.
        # Font size is bounded by BOTH constraints, and the width one is the
        # binding constraint on a wide table: sizing from row count alone gave a
        # font whose text was wider than the cells and matplotlib simply clipped
        # it, which is how the headers ended up reading "algorith" and "when".
        total_lines = sum(row_lines) + header_lines
        axes_width_pt = 0.90 * PAGE_SIZE[0] * 72.0
        axes_height_pt = (top - 0.05) * PAGE_SIZE[1] * 72.0
        # DejaVu Sans averages ~0.60 em per character; bold headers run wider,
        # hence the margin.
        by_width = axes_width_pt / (0.62 * max(sum(char_budget), 1)) * 0.94
        by_height = axes_height_pt / (1.5 * max(total_lines, 1))
        fontsize = max(self.TABLE_MIN_FONT,
                       min(self.TABLE_MAX_FONT, by_width, by_height))
        rendered.set_fontsize(fontsize)
        # Cap a text line's share of the page so a two-row table does not
        # stretch two rows over the whole sheet; only a long table divides the
        # page evenly.
        line_height = min(1.0 / max(total_lines, 1), 0.075)
        for (row, _), cell in rendered.get_celld().items():
            lines = header_lines if row == 0 else row_lines[row - 1]
            cell.set_height(line_height * lines)
            cell.set_edgecolor(st.GRID_COLOUR)
            cell.set_linewidth(0.5)
            cell.PAD = 0.04
            cell.get_text().set_verticalalignment("center")
            if row == 0:
                cell.set_text_props(weight="bold", color=st.TEXT_PRIMARY)
                cell.set_facecolor("#f2f1ee")

        if len(payload["table"]) > payload["max_rows"]:
            fig.text(0.05, 0.02,
                     f"showing {payload['max_rows']} of "
                     f"{len(payload['table'])} rows — full table in "
                     f"tables/{payload['name']}.csv",
                     fontsize=8, color=st.TEXT_SECONDARY)
        return fig


def _layout(table, char_budget: int):
    """Share the page between columns, then wrap each cell to its share.

    Returns (column width fractions, wrapped cell text, lines per row).

    Columns are sized by the width they actually need -- the longest word in the
    header or body -- and the surplus is shared among them in proportion to
    their natural length, with the deliberately-wide columns (paths, messages)
    capped so one of them cannot squeeze out every column a reader compares.
    Nothing is truncated: whatever does not fit on a line wraps onto the next.
    """
    import textwrap

    columns = [str(c) for c in table.columns]
    values = [[str(v) for v in row] for row in table.values.tolist()]

    natural, minimum = [], []
    for i, name in enumerate(columns):
        cells = [name] + [row[i] for row in values]
        natural.append(max(len(c) for c in cells))
        # A column can never be narrower than its own HEADER -- squeezing one
        # below that is what made "algorithm" and "events" run into each other.
        # It is NOT floored by the widest word in the BODY: a filesystem path is
        # a single unbreakable 60-character word, and treating that as a floor
        # blew the column past the page budget, which is how paths ended up
        # running out of their cell and over the next column. `wrap` sets
        # break_long_words, so a body word that does not fit is split instead.
        minimum.append(max(len(name), 4))

    capped = [min(n, 40 if columns[i] in table_io.WIDE_COLUMNS else 22)
              for i, n in enumerate(natural)]
    floors = list(minimum)
    budget = [max(c, f) for c, f in zip(capped, floors)]

    total = sum(budget)
    if total > char_budget:
        # Shrink proportionally, but never below a column's unbreakable floor.
        scale = char_budget / total
        budget = [max(int(b * scale), f) for b, f in zip(budget, floors)]
        total = sum(budget)

    widths = [b / total for b in budget]

    def wrap(text, width):
        # break_long_words so an unbroken path still fits its column.
        return "\n".join(textwrap.wrap(text, width=max(width, 4),
                                       break_long_words=True,
                                       break_on_hyphens=True)) or ""

    headers = [wrap(name, budget[i]) for i, name in enumerate(columns)]
    header_lines = max((h.count("\n") + 1 for h in headers), default=1)

    wrapped, row_lines = [], []
    for row in values:
        cells, lines = [], 1
        for i, cell in enumerate(row):
            text = wrap(cell, budget[i])
            cells.append(text)
            lines = max(lines, text.count("\n") + 1)
        wrapped.append(cells)
        row_lines.append(lines)
    return widths, wrapped, row_lines, headers, header_lines, budget


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in text]
    slug = "".join(keep)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")[:60]
