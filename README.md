[![DOI](https://zenodo.org/badge/273115663.svg)](https://zenodo.org/badge/latestdoi/273115663)
![Unit test workflow](https://github.com/UWB-Biocomputing/Graphitti/workflows/Unit%20Tests/badge.svg)
[![Check for Code Style Violations](https://github.com/UWB-Biocomputing/Graphitti/actions/workflows/format.yml/badge.svg)](https://github.com/UWB-Biocomputing/Graphitti/actions/workflows/format.yml)

# Graphitti

## About

Graphitti is a high-performance simulator of graph based systems, currently being applied to
computational neuroscience and emergency communications systems. Graphitti supports vertices and
edges with internal state, message passing between vertices over edges, graph architecture changes
(edge creation and destruction), vertex spatial locations (currently (_x_, _y_), but with the
possibility of adding _z_), multiple vertex types, and a flexible system for recording data as
simulations progress.

Graphitti runs on both CPUs and GPUs and can simulate very large graphs (tens of thousands of
vertices; hundreds of thousands to millions of edges) for long durations (billions of time steps).


## Code of Conduct
This project and everyone participating in it is governed by the [Intelligent Networks Laboratory Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Contributing
Please refer to the [Graphitti Project Contributing Guide](CONTRIBUTING.md) for information about
how internal and external contributors can work on Graphitti.
