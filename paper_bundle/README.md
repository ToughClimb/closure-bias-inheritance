# Paper Draft

This directory contains a self-contained LaTeX manuscript draft suitable for packaging.

## Files

- `main.tex`: manuscript entry point
- `sections/`: section-level source files
- `figures/`: figures referenced by the manuscript
- `artifacts/`: CSV/markdown summaries used to populate the tables and cross-check results
  - `artifacts/source/`: raw benchmark CSV outputs used to build the summaries and tables

## Current status

- The draft is aligned with the current benchmark results and theory notes.
- Figures are stored locally under `figures/` (no `../results/...` paths).
- Benchmark summaries used for consistency checks are copied under `artifacts/`.
- The recommended build tool is `tectonic` (via your system install).

## Build

Run:

```bash
./paper_bundle/build.sh
```

To build the Elsevier/JCP-style entry (`elsarticle`):

```bash
./paper_bundle/build.sh main_elsarticle.tex
```

or manually (if `tectonic` is on your `PATH`):

```bash
tectonic --outdir paper_bundle/build paper_bundle/main.tex
```

## Suggested next writing steps

1. Refine the introduction and theory wording for the target venue.
2. Add citations and a bibliography once the target venue is fixed.
3. Extend the appendix if additional defensive experiments are promoted into the main paper.
