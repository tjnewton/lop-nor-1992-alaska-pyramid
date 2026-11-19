# Could seismic waves from the 1992 Lop Nor nuclear explosion have revealed a pyramid buried in Alaska?

Code and tables for the paper of that title (Tyler J. Newton, submitted to *Seismica* as a Report; preprint: *to be added*). A popular story holds that seismic waves from the 21 May 1992 Lop Nor explosion revealed a pyramid buried in Alaska. The paper tests whether any recording of that explosion could have shown such a body, and returned its place, depth or shape.

This repository holds what is needed to reproduce the paper's numbers and figures: the analysis code, the tables behind every number, the figure script and the software environment. Archived at Zenodo: *DOI to be added at the first release*.

## Layout

| Path | What it holds |
|---|---|
| `akp/setup/` | The event and its bulletin readings, station metadata, and the first probes of the open data. |
| `akp/stage_a/` | Stations of 1992 (`a1_*`); coda amplitudes of the 1992 record and of later explosions (`a2_*`, `a5_*`, including the late windows of `a5_late_windows.py`); the simulations (`a8_*`: the base cases, the detection-favorable cases, the velocity-gradient case) and their accuracy checks; the array runs at Eielson (`a7_*`) and Eskdalemuir (`a9_*`); the estimate for a base wider than simulated (`a5_late_base.py`) and the comparison with the regional Q of crustal waves (`a5_late_regional_q.py`). |
| `akp/stage_p2/`, `akp/stage_p3/` | The terrain screen and the airborne magnetic and electromagnetic search (Supplement Two). |
| `akp/stage_p4/` | The inventory of receivers (`s9_*`, `s10_*`), the real-mound check of the terrain screen (`t1_*`), the stations by version of the claim with the bounds beyond the simulated range (`s11_versions.py`), and the sources' miles and feet in SI units (`s12_units.py`). |
| `figures/make_figures.py` | Every figure of the paper and its supplements. |
| `provenance/numbers.csv` | For each number of the paper taken from a table: where it is printed, the printed value, its table, column and rows. |
| `provenance/check_numbers.py` | Recomputes each of those numbers from the tables and compares it with the printed value. |

Each module states in its docstring its inputs, its method, its outputs and how to run it. Its tables are written beside it.

## Setup

1. Install conda (for example Miniforge), then from the repository root run `make env-akp`. This creates the environment `swe-akp` from `environment-akp.yml` and installs the seismology toolkit `seismic-workflow-engine` at the commit the results used. `env-akp.lock.txt` records the exact packages of the environment that produced the results (macOS, arm64).
2. Set `AKP_DATA` to a folder with room for the data that the modules fetch and cache, for example `export AKP_DATA=/path/to/akp-data`. Fetching everything takes about 200 GB: about 125 GB of terrain tiles and 50 GB of airborne line data for Supplement Two; the seismological parts need under 3 GB.
3. Run modules from the repository root, as their docstrings show, for example `PYTHONPATH=. python -m akp.stage_p4.s11_versions`. `scripts/mem_guard.py` wraps the heavy runs with a memory limit.

## Checking the numbers

`make check` (or `python provenance/check_numbers.py`) recomputes every number that the paper and its supplements take from a table, from the tables in this repository, and reports any that differ from the printed value. It needs only the environment, not the data.

## Data

All inputs are open. The modules fetch them and cache them under `AKP_DATA`:

- the ISC Bulletin and the International Registry of Seismograph Stations (International Seismological Centre);
- waveforms and station metadata from EarthScope, for the networks the paper cites with their FDSN DOIs;
- the Raspberry Shake station service (network AM);
- receiver positions from USGS data reports and the EarthScope assembled archive;
- the USGS IfSAR terrain model of Alaska; the statewide geologic, permafrost and land-cover maps; the Alaska vent database; the USGS airborne survey inventory and its line data; the Alaska Resource Data File; a global model of depth to bedrock.

The paper gives the full references. Licensed news material and podcast audio are cited in the paper and not included.

## Things to know

- **The simulations are heavy.** The largest of the eighteen three-dimensional runs took about five and a half hours and 13 GB of memory on the laptop used. Their measured tables are included, so the numbers and figures do not need the runs.
- **Freeze guards.** The stages of Supplement Two that read the screened area check, before they open its data, that their code is unchanged since its methods were fixed. That check reads the version history of the original working repository, which is not public, so these stages stop with a message in this repository. Their results are in the tables.
- **Raspberry Shake stations** stand at private hosts' homes, and their data may not be redistributed. In `akp/stage_p4/s9_receivers.csv` their positions are rounded to 0.1 degree, their codes withheld, their position uncertainty and distances to the search grid left blank, and their rows placed at the end of the table.
- **Comments in the code** name the stages of the project (for example A1 to A9, P2 to P4 and Track A) and the versions of the claim by their labels C1 to C10, as in the timeline figure. The rules behind the paper's claims are summarized in its Supplement One, section S1.10.
- **Changes from the code that produced the results:** comments, docstrings, messages and table notes edited to remove references to the project's internal records (the build compares the syntax tree of every module before and after, with strings blanked, and allows only the changes listed here); in code, one command-line flag renamed (`--full-size`) and five paths of files that are not public removed from the lists of two freeze guards; two unused columns dropped from the input table of the timeline figure; the paths of the figure script adapted to this layout; thirteen tables that nothing reads left out; the data manifest's rows for the licensed claim sources left out; and the Raspberry Shake changes above.

## License and citation

Code: MIT (`LICENSE`). Tables and figures: CC BY 4.0, with the third-party terms in `LICENSE-DATA.md`. Please cite the paper and this archive; `CITATION.cff` has the details.
