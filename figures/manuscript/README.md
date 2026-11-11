# Manuscript figures (Figs. 1-6)

All figures are drawn at the journal's double-column width (174 mm) with lettering of 8-10 pt.
Run any script from this folder; the PDF and a 600-dpi PNG are written to `out/`.

| Script | Figure |
|--------|--------|
| `fig1_overview.py` | Fig. 1, study overview |
| `fig2_examples.py` | Fig. 2, example randomized crops (CBIS-DDSM and DMID); the crop tiles are read from `fig2_tiles.png` |
| `fig3_architecture.py` | Fig. 3, architecture and block options |
| `figs4to6_results.py` | Figs. 4-6, model levels, paired contrasts and code replacement, from `results/five_seed/stats` (`rev5_levels.csv`, `rev5_contrasts.csv`, `rev5_cbis_drops.csv`) |

Requirements: matplotlib, numpy, pandas, Pillow and the Liberation Sans font.
