# Cluster Point Process Input Generator

`cluster_point_process.py` generates synthetic 911 emergency-call input for
Graphitti simulations. Calls are modeled as a spatio-temporal cluster point
process: a primary incident process in time and space, with secondary calls
clustered around each incident.

Output is a Graphitti `simulator_inputs` XML file (see
[Graph and Event Inputs](https://uwb-biocomputing.github.io/Graphitti/Developer/GraphAndEventInputs.html)).

## Requirements

- Python 3
- `numpy`, `networkx`, `lxml`, `pandas`
- `PyQt5` (GUI mode only; not required for `--config` or `--write-config`)

```bash
pip install numpy networkx lxml pandas PyQt5
```

Run from this directory so `cluster_point_process_functions` imports correctly.

## Usage

### Graphical interface (default)

```bash
python3 cluster_point_process.py
```

Fill in the graph file, timing parameters, type ratios, and prototypes, then
click **Generate Events**.

### Create a boilerplate JSON config

Defaults live in the version-controlled template
[`params.example.json`](params.example.json) (aligned with NG911 regression
documentation). Copy it to your own config with:

```bash
python3 cluster_point_process.py --write-config
# copies params.example.json to params.json in the current directory

python3 cluster_point_process.py --write-config my_run.json
```

You can also copy the template manually: `cp params.example.json params.json`.

Edit the file (graph path, `graph_id`, timing, ratios, prototypes), then run
headless:

```bash
python3 cluster_point_process.py --config params.json
```

Relative paths for `graph_file` and `output_path` in the JSON file are resolved
from the config file's directory.

## Command-line options

| Option | Description |
|--------|-------------|
| *(none)* | Open the PyQt GUI. |
| `--write-config [FILE.json]` | Copy `params.example.json` to `FILE.json` (default: `params.json`) and exit. |
| `--config FILE.json` | Load parameters from JSON, generate XML without the GUI, and exit. |

## JSON config parameters

### Required

| Key | Type | Description |
|-----|------|-------------|
| `graph_file` | string | Path to a `.graphml` graph file. |
| `graph_id` | string or number | Node id of the caller-region vertex in that graph. |
| `first` | number | Start time for primary events (seconds). |
| `last` | number | End time for primary events (seconds). |
| `mean_time_interval` | number | Mean interval between primary events (seconds). |
| `dead_time_after_event` | number | Dead time after each primary event (seconds). |
| `mean_call_interval_after_incident` | number | Mean interval between incident and follow-up calls (seconds). |
| `mean_duration` | number | Mean call duration (seconds). |
| `minimum_duration` | number | Minimum call duration (seconds). |
| `mean_patience_time` | number | Mean caller patience time (seconds). |
| `mean_on_site_time` | number | Mean on-site responder time (seconds). |
| `type_ratios` | object | Call-type probabilities (e.g. `Law`, `EMS`, `Fire`). Must sum to 1.0 within ±0.02; values are normalized if slightly off. |
| `prototypes` | object | Prototype definitions keyed by id (`"0"` … `"3"`). Each entry includes spatial/intensity parameters and an optional `weight`. |

Each prototype object uses these fields:

| Field | Description |
|-------|-------------|
| `mu_r` | Mean cluster radius. |
| `sdev_r` | Standard deviation of cluster radius. |
| `mu_intensity` | Mean cluster intensity (drives secondary call count). |
| `sdev_intensity` | Standard deviation of cluster intensity. |
| `weight` | Optional relative selection weight. If omitted for all prototypes and exactly four prototypes are defined, legacy 40/50/9/1% weights apply; otherwise selection is uniform. Weights need not sum to 1 (they are normalized). |

### Optional

| Key | Default | Description |
|-----|---------|-------------|
| `random_seed` | *(none)* | Integer seed for `numpy.random`. Omit for non-deterministic runs. |
| `output_path` | `<GRAPH_BASENAME>_cluster_point_process.xml` in the current working directory | Path for the output XML. |
| `clock_tick_size` | `"1"` | Written to the output XML `data` element. |
| `clock_tick_unit` | `"sec"` | Written to the output XML `data` element. |

### Example

```json
{
  "graph_file": "../../../../configfiles/graphs/test-medium-911.graphml",
  "graph_id": "4",
  "first": 34,
  "last": 3600,
  "mean_time_interval": 62.88,
  "dead_time_after_event": 1,
  "mean_call_interval_after_incident": 20,
  "mean_duration": 204,
  "minimum_duration": 4,
  "mean_patience_time": 50,
  "mean_on_site_time": 1200,
  "type_ratios": { "Law": 0.33, "EMS": 0.33, "Fire": 0.34 },
  "prototypes": {
    "0": {
      "mu_r": 0.0005,
      "sdev_r": 0.0001,
      "mu_intensity": 500000,
      "sdev_intensity": 50000,
      "weight": 0.4
    }
  },
  "random_seed": 42,
  "output_path": "output/cluster_point_process.xml"
}
```

See [`params.example.json`](params.example.json) for the full starter config.
It includes all four default prototypes with values aligned to the NG911
regression documentation (`docs/Developer/RegressionTestsDocumentation.md`).
Adjust `last` and other tuning fields for your scenario.

## Output

The script prints progress to stdout and writes an XML file. Each `event` under
the selected caller-region `vertex` includes time, duration, coordinates, call
type, patience, and on-site time.

## Related files

- `params.example.json` — version-controlled boilerplate config template
- `cluster_point_process_functions.py` — primary/secondary process logic
- `../README.md` — overview of all input-generation tools
