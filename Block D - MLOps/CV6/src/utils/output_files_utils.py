"""Pdf and CSV generation utilities for root analysis reports."""

import csv
import io
import os
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import MaxNLocator


def build_root_measurement_csv(entries: List[Dict[str, float]]) -> Tuple[str, bytes]:
    """Convert a list of per-plant measurement dicts into a CSV file (as bytes).

    Each *entry* should have the keys::

        filename   plant   length   bottom_tip   top_tip
        smoothness   angle   depth   span

    where ``bottom_tip`` and ``top_tip`` are ``(row, col)`` tuples or *None*.

    Args:
        entries: List of dictionaries, one per plant measurement.

    Returns:
        Tuple containing:
            * filename: Always ``'root_summary.csv'``.
            * file_bytes: UTF-8 encoded bytes of the CSV content, ready for
              saving or sending in an HTTP response.
    """
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)

    # Header row
    writer.writerow(
        [
            "filename",
            "plant",
            "length_px",
            "bottom_tip_y",
            "bottom_tip_x",
            "top_tip_y",
            "top_tip_x",
            "smoothness",
            "angle_deg",
            "depth",
            "span",
        ]
    )

    for row in entries:
        writer.writerow(
            [
                row["filename"],
                row["plant"],
                row["length"],
                row["bottom_tip"][0] if row["bottom_tip"] else None,
                row["bottom_tip"][1] if row["bottom_tip"] else None,
                row["top_tip"][0] if row["top_tip"] else None,
                row["top_tip"][1] if row["top_tip"] else None,
                row["smoothness"],
                row["angle"],
                row["depth"],
                row["span"],
            ]
        )

    return ("root_summary.csv", csv_buf.getvalue().encode("utf-8"))


# ─── Report Generation Function ───────────────────────
def generate_root_analysis_report(
    csv_path: str,
    output_path: str = "full_analysis_report.pdf",
    pink: str = "#ED1E79",
    title: str = "IRIS Analysis Report",
    icon_path: str = r"CV6\src\app\iris_logo.png",
    font_family: str = "DejaVu Sans",
    chunk_size: int = 20
) -> None:
    """Generates a multi-page PDF report from a root measurement CSV."""
    # ─── Local Configuration ───────────────────────────

    PINK = pink
    TITLE = title
    ICON = icon_path
    FONT = font_family
    CHUNK_SIZE = chunk_size

    # ─── Load & Normalize Data ─────────────────────────

    df_all = pd.read_csv(csv_path)
    df_all = df_all.rename(columns={"length_px": "length", "angle_deg": "angle"})
    df = df_all[df_all["length"] > 0].copy()

    # ─── Validate Required Columns ──────────────────────

    required_cols = [
        "filename",
        "plant",
        "length",
        "bottom_tip_y",
        "bottom_tip_x",
        "top_tip_y",
        "top_tip_x",
        "smoothness",
        "angle",
        "depth",
        "span",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # ─── Compute Per-Image Summary (preserve CSV order) ──────────────────────
    # Capture original filename order
    original_order = df["filename"].drop_duplicates().tolist()
    summary = (
        df.groupby("filename", sort=False)
        .agg(
            Plant_Count=("plant", "count"),
            Avg_Length=("length", "mean"),
            Avg_Angle=("angle", "mean"),
            Avg_Depth=("depth", "mean"),
            Avg_Smoothness=("smoothness", "mean"),
            Avg_Span=("span", "mean"),
        )
        .round(2)
        .reset_index()
    )
    # Reorder summary to match CSV appearance
    summary = summary.set_index("filename").loc[original_order].reset_index()

    # ─── Generate PDF Report ───────────────────────────
    with PdfPages(output_path) as pdf:

        # ─── Page 1: Title & Metadata ──────────────────
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")

        # Header ax: logo + title on same line
        axh = fig.add_axes([0, 0.88, 1, 0.12])
        axh.axis("off")

        # Embed logo via AnnotationBbox, increased size
        if os.path.exists(ICON):
            img = mpimg.imread(ICON)
            imagebox = OffsetImage(img, zoom=0.3)
            ab = AnnotationBbox(
                imagebox, (0.11, 0.45), frameon=False, xycoords="axes fraction"
            )
            axh.add_artist(ab)

        # Draw title centered on same axis, larger font
        axh.text(
            0.5,
            0.5,
            TITLE,
            va="center",
            ha="center",
            fontsize=32,
            fontweight="bold",
            color=PINK,
            family=FONT,
        )

        # Subheader ax: metadata
        axm = fig.add_axes([0, 0.88, 1, 0.03])
        axm.axis("off")
        meta = f"Date: {datetime.now():%Y-%m-%d}    Images: {summary.shape[0]}"
        axm.text(0.5, 0.5, meta, ha="center", va="center", fontsize=10, family=FONT)

        # Separator before insights
        sep0 = Line2D(
            [0.05, 0.95],
            [0.88, 0.88],
            transform=fig.transFigure,
            color=PINK,
            linewidth=1,
        )
        fig.add_artist(sep0)

        # ─── Page 1: Key Insights ──────────────────────
        fig.text(
            0.05,
            0.85,
            "Key Insights",
            ha="left",
            va="center",
            fontsize=16,
            fontweight="bold",
            color=PINK,
            family=FONT,
        )

        # Bullets block: pink bullets, black text, no inline number coloring
        lmin, lmax = df["length"].min(), df["length"].max()
        alow, ahigh = df["angle"].quantile(0.1), df["angle"].quantile(0.9)
        dmin, dmax = df["depth"].min(), df["depth"].max()
        span_avg = df["span"].mean()
        smo_avg = df["smoothness"].mean()
        sentences = [
            f"Across all samples, root lengths ranged from {lmin:.1f}px"
            f" to {lmax:.1f}px.",
            f"Most roots exhibited growth angles between {alow:.1f}° and {ahigh:.1f}°.",
            f"Root depths spanned from {dmin:.1f}px to {dmax:.1f}px.",
            f"The average lateral span was {span_avg:.1f}px.",
            f"Mean smoothness was {smo_avg:.2f}, indicating overall straightness.",
        ]
        y = 0.815

        for sentence in sentences:
            # draw pink bullet
            fig.text(0.05, y, "•", color=PINK, fontsize=10, family=FONT)
            # draw full sentence in black, expanding as needed
            fig.text(0.07, y, sentence, color="black", fontsize=10, family=FONT)
            y -= 0.025

        # Separator after bullets
        sep1 = Line2D(
            [0.05, 0.95], [y, y], transform=fig.transFigure, color=PINK, linewidth=1
        )
        fig.add_artist(sep1)

        # ─── Page 1: Mini Metric Distributions (2×3 grid) ────
        gs = fig.add_gridspec(
            3,
            2,
            left=0.085,
            right=0.95,
            bottom=0.05,
            top=y - 0.03,
            hspace=0.4,
            wspace=0.2,
        )
        axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(6)]
        dist_cols = [
            "Avg_Length",
            "Avg_Angle",
            "Avg_Depth",
            "Avg_Smoothness",
            "Avg_Span",
        ]

        # Five histograms
        for i, metric in enumerate(dist_cols):
            axd = axes[i]
            axd.hist(summary[metric], bins=10, color=PINK, edgecolor="black")
            # Title bold
            axd.set_title(
                metric.replace("Avg_", "Average "), fontsize=10, fontweight="bold"
            )
            # X-axis ticks tilt
            axd.tick_params(axis="x", rotation=35)
            # Y-axis label and integer ticks
            axd.set_ylabel("Number of Images", fontsize=8)
            axd.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Sixth slot: Plant count per image bar chart
        axd = axes[5]
        # Compute integer counts per plant count
        counts = summary["Plant_Count"].value_counts().sort_index()
        # Ensure x-axis uses integer tick labels
        x_vals = counts.index.astype(int)
        axd.bar(x_vals, counts.values, color=PINK, edgecolor="black")
        axd.set_title("Plant Count per Image", fontsize=10, fontweight="bold")
        axd.set_xlabel("Number of Plants", fontsize=8)
        axd.set_ylabel("Number of Images", fontsize=8)
        axd.set_xticks(x_vals)
        axd.tick_params(axis="x", rotation=35)

        # Save Page 1
        pdf.savefig(fig)
        plt.close(fig)

        # ─── Summary Table Pages ────────────────────────────
        for start in range(0, len(summary), CHUNK_SIZE):
            chunk = summary.iloc[start : start + CHUNK_SIZE]
            fig_tab, ax_tab = plt.subplots(figsize=(8.27, 11.69))
            ax_tab.axis("off")

            # Prepare table data
            data = [
                [
                    r["filename"],
                    r["Plant_Count"],
                    f"{r['Avg_Length']:.1f}",
                    f"{r['Avg_Angle']:.1f}",
                    f"{r['Avg_Depth']:.1f}",
                    f"{r['Avg_Smoothness']:.2f}",
                    f"{r['Avg_Span']:.1f}",
                ]
                for _, r in chunk.iterrows()
            ]

            # Column width: filename wider (28%), others share 72%
            colWidths = [0.28] + [0.72 / 6] * 6

            # Dynamically size table height based on rows (including header)
            n_rows = len(data) + 1
            row_height = 0.04
            table_height = min(row_height * n_rows, 0.90)
            table_bottom = 1.0 - table_height

            # Draw table
            table = ax_tab.table(
                cellText=data,
                colLabels=[
                    "Filename",
                    "Count",
                    "Length",
                    "Angle",
                    "Depth",
                    "Smoothness",
                    "Span",
                ],
                colWidths=colWidths,
                cellLoc="center",
                colLoc="center",
                bbox=[0.01, table_bottom, 0.98, table_height],
            )

            table.auto_set_font_size(False)
            table.set_fontsize(9)
            for (__, __), cell in table.get_celld().items():
                if __ == 0:
                    cell.set_facecolor(PINK)
                    cell._text.set_color("white")
                else:
                    cell.set_facecolor("white")
                    cell._text.set_color("black")

            # Fixed row heights
            table.scale(1, 1.2)

            # Enable wrapping in filename column
            for (_row, col), cell in table.get_celld().items():
                if col == 0:
                    cell._text.set_wrap(True)

            pdf.savefig(fig_tab)
            plt.close(fig_tab)

        # ─── Graph Pages: Visual Summaries ─────────────────
        fig4 = plt.figure(figsize=(8.27, 11.69))
        fig4.patch.set_facecolor("white")

        # Title + separator
        fig4.text(
            0.5,
            0.96,
            "Visual Summaries",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color=PINK,
            family=FONT,
        )

        sep = Line2D(
            [0.05, 0.95],
            [0.94, 0.94],
            transform=fig4.transFigure,
            color=PINK,
            linewidth=1,
        )
        fig4.add_artist(sep)

        # 3×2 grid under the title
        gs4 = fig4.add_gridspec(
            3, 2, left=0.1, right=0.95, top=0.9, bottom=0.08, hspace=0.4, wspace=0.3
        )
        ax = [fig4.add_subplot(gs4[i // 2, i % 2]) for i in range(6)]

        # 1) Histogram: all root lengths
        ax[0].hist(df["length"], bins=15, color=PINK, edgecolor="black")
        ax[0].set_title("Root Length Distribution", fontsize=10, fontweight="bold")
        ax[0].set_xlabel("Length (px)", fontsize=8)
        ax[0].set_ylabel("Count", fontsize=8)
        ax[0].yaxis.set_major_locator(MaxNLocator(integer=True))

        # 2) Histogram: all root angles
        ax[1].hist(df["angle"], bins=15, color=PINK, edgecolor="black")
        ax[1].set_title("Root Angle Distribution", fontsize=10, fontweight="bold")
        ax[1].set_xlabel("Angle (°)", fontsize=8)
        ax[1].set_ylabel("Count", fontsize=8)
        ax[1].yaxis.set_major_locator(MaxNLocator(integer=True))

        # 3) Scatter: Length vs. Smoothness
        ax[2].scatter(df["length"], df["smoothness"], color=PINK, alpha=0.8)
        ax[2].set_title("Length vs. Smoothness", fontsize=10, fontweight="bold")
        ax[2].set_xlabel("Length (px)", fontsize=8)
        ax[2].set_ylabel("Smoothness", fontsize=8)

        # 4) Scatter: Span vs. Depth
        ax[3].scatter(df["span"], df["depth"], color=PINK, alpha=0.8)
        ax[3].set_title("Span vs. Depth", fontsize=10, fontweight="bold")
        ax[3].set_xlabel("Span (px)", fontsize=8)
        ax[3].set_ylabel("Depth (px)", fontsize=8)

        # 5) IQR Outlier Highlight for Angle
        q1, q3 = df["angle"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        ax[4].scatter(df.index, df["angle"], color=PINK, alpha=0.8)
        out = df[(df["angle"] < lower) | (df["angle"] > upper)]
        ax[4].scatter(out.index, out["angle"], color="red")
        ax[4].axhline(lower, linestyle="--", color="purple")
        ax[4].axhline(upper, linestyle="--", color="purple")
        ax[4].set_title("Angle Outliers", fontsize=10, fontweight="bold")
        ax[4].set_xlabel("Sample Index", fontsize=8)
        ax[4].set_ylabel("Angle (°)", fontsize=8)

        # 6) Correlation heatmap of summary metrics
        # build the corr matrix as before
        corr = summary[
            [
                "Plant_Count",
                "Avg_Length",
                "Avg_Angle",
                "Avg_Depth",
                "Avg_Smoothness",
                "Avg_Span",
            ]
        ].corr()

        # define your “pretty” short names
        short = {
            "Plant_Count": "Count",
            "Avg_Length": "Length",
            "Avg_Angle": "Angle",
            "Avg_Depth": "Depth",
            "Avg_Smoothness": "Smoothness",
            "Avg_Span": "Span",
        }

        # rename the index & columns
        corr = corr.rename(index=short, columns=short)

        # pick your pink diverging cmap
        pink_div = sns.diverging_palette(345, 15, s=75, l=50, as_cmap=True)

        # draw it
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=pink_div, cbar=False, ax=ax[5])

        # finally, rotate and pad the tick labels so they never overflow
        ax[5].set_xticklabels(
            ax[5].get_xticklabels(), rotation=45, ha="right", fontsize=8
        )
        ax[5].set_yticklabels(ax[5].get_yticklabels(), rotation=45, fontsize=8)
        ax[5].set_title("Summary Metrics Correlation", fontsize=10, fontweight="bold")

        # Final layout & save
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig4)
        plt.close(fig4)
        # ─── Finalize PDF ───────────────────────────────
