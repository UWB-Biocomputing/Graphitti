# Import necessary libraries
import argparse
import ast
import json
import os
import shutil
import sys
import time

import lxml.etree as et
import networkx as nx
import numpy as np


def _is_headless_cli_run():
    """Return True when the process should run without importing the PyQt GUI."""
    headless_flags = ("--config", "--write-config")
    return any(flag in sys.argv for flag in headless_flags) or any(
        arg.startswith(f"{flag}=") for flag in headless_flags for arg in sys.argv[1:]
    )


if _is_headless_cli_run():
    _DialogBase = object
    _WidgetBase = object
else:
    from PyQt5.QtWidgets import (
        QApplication,
        QWidget,
        QLabel,
        QLineEdit,
        QPushButton,
        QVBoxLayout,
        QFileDialog,
        QMessageBox,
        QDialog,
        QDialogButtonBox,
        QGridLayout,
    )

    _DialogBase = QDialog
    _WidgetBase = QWidget

from cluster_point_process_functions import (
    DEFAULT_LEGACY_PROTOTYPE_WEIGHTS,
    primprocess,
    add_types,
    secprocess,
    add_vertex_events,
)

# source venv/bin/activate
# python3 cluster_point_process.py
# python3 cluster_point_process.py --write-config params.json
# python3 cluster_point_process.py --config params.json

# This script provides a GUI to configure and generate synthetic 911 call data using a cluster point process model.
# It integrates primary and secondary event generation with user-configurable parameters.
# All simulation-tuned values are supplied via the UI or a JSON file (--config); no graph path or RNG seed is fixed in code.


GRAPH_FILE_FIELD = "Select Graph File (.graphml):"
SEED_FIELD_LABEL = "Random seed (optional, blank to use current NumPy RNG state):"
BOILERPLATE_CONFIG_TEMPLATE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "params.example.json"
)

_TYPE_RATIO_SUM_TOLERANCE = 0.02


def _validate_and_normalize_type_ratios(type_ratios):
    """Coerce type_ratios values to float, validate range/sum, and normalize to sum 1."""
    coerced = {}
    ratio_sum = 0.0
    for ratio_key, ratio_value in type_ratios.items():
        try:
            ratio = float(ratio_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"type_ratios[{ratio_key!r}] must be a numeric probability "
                f"(got {ratio_value!r})"
            ) from exc

        if not np.isfinite(ratio):
            raise ValueError(
                f"type_ratios[{ratio_key!r}] must be finite (got {ratio_value!r})"
            )

        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(
                f"type_ratios[{ratio_key!r}] must be between 0.0 and 1.0 "
                f"(got {ratio:g})"
            )

        coerced[ratio_key] = ratio
        ratio_sum += ratio

    if ratio_sum <= 0.0:
        raise ValueError("type_ratios must have a positive sum")

    if abs(ratio_sum - 1.0) > _TYPE_RATIO_SUM_TOLERANCE:
        raise ValueError(f"type_ratios should sum to 1.0 (got {ratio_sum:g})")

    if abs(ratio_sum - 1.0) > 1e-12:
        coerced = {k: v / ratio_sum for k, v in coerced.items()}

    return coerced


def _parse_graph_segments(segments_attr):
    """Parse a graph node's segments attribute into an (n, 2, 2) region grid."""
    try:
        parsed = ast.literal_eval(segments_attr)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(
            "Graph node 'segments' must be a literal list of bounding boxes, "
            f"not executable code: {exc}"
        ) from exc

    graph_grid = np.asarray(parsed, dtype=float)
    if graph_grid.ndim != 3 or graph_grid.shape[1:] != (2, 2):
        raise ValueError(
            "Graph node 'segments' must be a list of bounding boxes with shape "
            f"(n, 2, 2); got array shape {graph_grid.shape}"
        )
    if graph_grid.size == 0:
        raise ValueError("Graph node 'segments' must not be empty")
    return graph_grid


def _coerce_prototype_keys(prototypes, prototype_weights):
    """Normalize string digit keys (e.g. from JSON) to integers when every key is numeric."""
    if prototypes and all(isinstance(k, str) and k.isdigit() for k in prototypes):
        prototypes = {int(k): v for k, v in prototypes.items()}
    if prototype_weights and all(isinstance(k, str) and k.isdigit() for k in prototype_weights):
        prototype_weights = {int(k): float(v) for k, v in prototype_weights.items()}
    return prototypes, prototype_weights


def generate_cluster_point_process_xml(
    graph_file,
    graph_id,
    first,
    last,
    mu,
    pp_dead_t,
    sec_proc_sigma,
    duration_mean,
    duration_min,
    patience_mean,
    onsite_mean,
    type_ratios,
    prototypes,
    prototype_weights,
    random_seed=None,
    output_path=None,
    clock_tick_size="1",
    clock_tick_unit="sec",
):
    """Run primary and secondary processes and write the Graphitti simulator_inputs XML.

    Parameters mirror the GUI fields. prototype_weights maps each prototype key to a
    relative frequency (need not sum to exactly 1; values are normalized). If None,
    legacy 40/50/9/1% weights apply when there are exactly four prototypes; otherwise
    selection is uniform.

    random_seed: if None or empty string, the RNG is not re-seeded.

    output_path: full path to the output XML; if None, ``<GRAPH_BASENAME>_cluster_point_process.xml``
    in the current working directory.
    """
    graph_file = os.path.abspath(os.path.expanduser(graph_file))
    if not os.path.isfile(graph_file):
        raise FileNotFoundError(f"Graph file not found: {graph_file}")

    prototypes, prototype_weights = _coerce_prototype_keys(prototypes, prototype_weights)
    if prototype_weights is not None and len(prototype_weights) == 0:
        prototype_weights = None

    type_ratios = _validate_and_normalize_type_ratios(type_ratios)

    if random_seed is not None and str(random_seed).strip() != "":
        np.random.seed(int(random_seed))

    graph = nx.read_graphml(graph_file)
    gid = str(graph_id)
    if gid not in graph.nodes:
        sample = list(graph.nodes)[:25]
        raise KeyError(
            f"Graph id {gid!r} not found in graph. Example node ids: {sample}"
        )

    graph_grid = _parse_graph_segments(graph.nodes[gid]["segments"])

    incidents = primprocess(first, last, mu, pp_dead_t, graph_grid)
    print(f"Number of Primary events: {incidents.shape[0]}")

    incidents_with_types = add_types(incidents, type_ratios)

    start_t = time.time()
    print("Generating Secondary events...")
    sec_events = secprocess(
        sec_proc_sigma,
        duration_mean,
        duration_min,
        patience_mean,
        onsite_mean,
        prototypes,
        incidents_with_types,
        prototype_weights=prototype_weights,
    )
    end_t = time.time()
    print("Elapsed time:", round(end_t - start_t, 4), "seconds")
    print("Number of Primary Events:", len(incidents_with_types))
    print("Number of Secondary Events:", sec_events.shape[0])

    graph_stem = os.path.splitext(os.path.basename(graph_file))[0].upper()
    if not output_path:
        output_path = graph_stem + "_cluster_point_process.xml"
    else:
        output_path = os.path.abspath(os.path.expanduser(output_path))

    inputs = et.Element("simulator_inputs")
    data = et.SubElement(
        inputs,
        "data",
        {
            "description": f"{graph_stem} Calls - Cluster Point Process",
            "clock_tick_size": str(clock_tick_size),
            "clock_tick_unit": str(clock_tick_unit),
        },
    )

    vertex_name = graph.nodes[gid]["name"]
    data = add_vertex_events(data, gid, vertex_name, sec_events)

    tree = et.ElementTree(inputs)
    tree.write(
        output_path,
        xml_declaration=True,
        encoding="UTF-8",
        pretty_print=True,
    )
    print("Secondary process was saved to:", output_path)
    return output_path


def write_boilerplate_json_config(output_path, overwrite=False):
    """Copy the repo template JSON config to output_path and return that path."""
    template_path = BOILERPLATE_CONFIG_TEMPLATE
    if not os.path.isfile(template_path):
        raise FileNotFoundError(
            f"Boilerplate template not found: {template_path}"
        )

    output_path = os.path.abspath(os.path.expanduser(output_path))
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing config: {output_path}. "
            "Pass overwrite=True or choose a different path."
        )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    shutil.copy2(template_path, output_path)
    print("Wrote boilerplate config to:", output_path)
    return output_path


def run_from_json_config(config_path):
    """Load parameters from JSON and generate XML without the GUI."""
    config_path = os.path.abspath(os.path.expanduser(config_path))
    config_dir = os.path.dirname(config_path)
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    graph_file = cfg["graph_file"]
    if not os.path.isabs(graph_file):
        graph_file = os.path.join(config_dir, graph_file)

    prototypes_cfg = cfg["prototypes"]
    prototypes = {}
    prototype_weights = {}
    for key, params in prototypes_cfg.items():
        if not isinstance(params, dict):
            raise TypeError(f'prototypes["{key}"] must be an object/dict')
        entry = dict(params)
        w = entry.pop("weight", None)
        prototypes[key] = entry
        if w is not None:
            prototype_weights[key] = float(w)
    if not prototype_weights:
        prototype_weights = None

    out = cfg.get("output_path")
    if out and not os.path.isabs(out):
        out = os.path.join(config_dir, out)

    generate_cluster_point_process_xml(
        graph_file=graph_file,
        graph_id=cfg["graph_id"],
        first=float(cfg["first"]),
        last=float(cfg["last"]),
        mu=float(cfg["mean_time_interval"]),
        pp_dead_t=float(cfg["dead_time_after_event"]),
        sec_proc_sigma=float(cfg["mean_call_interval_after_incident"]),
        duration_mean=float(cfg["mean_duration"]),
        duration_min=float(cfg["minimum_duration"]),
        patience_mean=float(cfg["mean_patience_time"]),
        onsite_mean=float(cfg["mean_on_site_time"]),
        type_ratios={k: float(v) for k, v in cfg["type_ratios"].items()},
        prototypes=prototypes,
        prototype_weights=prototype_weights,
        random_seed=cfg.get("random_seed"),
        output_path=out,
        clock_tick_size=str(cfg.get("clock_tick_size", "1")),
        clock_tick_unit=str(cfg.get("clock_tick_unit", "sec")),
    )


# ------------------------------
# Class: TypeRatioDialog
# ------------------------------
# Provides a dialog to input ratios for different 911 call types (Law, EMS, Fire).
class TypeRatioDialog(_DialogBase):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Set Type Ratios")
        self.init_ui()
        self.initial_values = {}  # Store initial values

    def init_ui(self):
        layout = QVBoxLayout()

        # Labels and input fields for call type ratios
        self.labels = ["Law", "EMS", "Fire"]
        self.entries = {}
        for label_text in self.labels:
            label = QLabel(label_text)
            entry = QLineEdit()
            self.entries[label_text] = entry

            layout.addWidget(label)
            layout.addWidget(entry)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.on_cancel_clicked)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self.backup_initial_values()  # Save initial values

    def accept(self):
        """Validates and processes user inputs when OK is clicked."""
        invalid_fields = []
        for label, entry in self.entries.items():
            text = entry.text().strip()
            if not text:
                invalid_fields.append(label)
            else:
                try:
                    float_val = float(text)
                    if float_val < 0:  # Ensure positive float numbers
                        invalid_fields.append(label)
                except ValueError:
                    invalid_fields.append(label)

        if invalid_fields:
            # Show an error message for invalid inputs
            error_message = "Invalid or empty values in the following fields:\n"
            for field in invalid_fields:
                error_message += f"- {field}\n"
            QMessageBox.warning(self, "Input Error", error_message)
        else:
            # Store validated results
            self.result = {
                label: float(entry.text().strip())
                for label, entry in self.entries.items()
            }
            ratio_sum = sum(self.result.values())
            if abs(ratio_sum - 1.0) > 0.02:
                QMessageBox.warning(
                    self,
                    "Input Error",
                    f"Type ratios should sum to 1.0 (currently {ratio_sum:g}).",
                )
                return
            super().accept()

    def backup_initial_values(self):
        """Stores initial values to restore them if the user cancels."""
        self.initial_values = {
            label: entry.text() for label, entry in self.entries.items()
        }

    def on_cancel_clicked(self):
        """Restores initial values if the dialog is canceled."""
        for label, entry in self.entries.items():
            current_text = entry.text().strip()
            if not current_text:  # If the field is empty, reset to initial value
                if label in self.initial_values:
                    entry.setText(self.initial_values[label])

        self.reject()  # Close the dialog


# ------------------------------
# Class: PrototypesDialog
# ------------------------------
# Allows the user to define prototype configurations for secondary event generation.
class PrototypesDialog(_DialogBase):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Set Prototypes")
        self.init_ui()

    def init_ui(self):
        """Sets up the layout and input fields for prototype configurations."""
        layout = QGridLayout()
        self.entries = {}

        # Labels for prototype parameters (spatial/intensity) plus relative selection weight
        labels = ["mu_r:", "sdev_r:", "mu_intensity:", "sdev_intensity:"]

        # Add input fields for four prototypes
        for i in range(4):
            prototype_label = QLabel(f"Prototype {i}:")
            layout.addWidget(prototype_label, i, 0)

            col = 1
            for label in labels:
                entry = QLineEdit()
                self.entries[f"Prototype {i} - {label}"] = entry
                layout.addWidget(QLabel(label), i, col)
                layout.addWidget(entry, i, col + 1)
                col += 2

            w_entry = QLineEdit()
            w_entry.setText(str(DEFAULT_LEGACY_PROTOTYPE_WEIGHTS[i]))
            self.entries[f"Prototype {i} - weight:"] = w_entry
            layout.addWidget(QLabel("weight:"), i, col)
            layout.addWidget(w_entry, i, col + 1)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box, 4, 0, 1, col + 2)

        self.setLayout(layout)

    def accept(self):
        """Validates and processes user inputs when OK is clicked."""
        invalid_fields = []
        for label, entry in self.entries.items():
            text = entry.text().strip()
            if not text:
                invalid_fields.append(label)
            else:
                try:
                    float_val = float(text)
                    if float_val < 0:  # Ensure positive float numbers
                        invalid_fields.append(label)
                except ValueError:
                    invalid_fields.append(label)

        if invalid_fields:
            # Show an error message for invalid inputs
            error_message = "Invalid or empty values in the following fields:\n"
            for field in invalid_fields:
                error_message += f"- {field}\n"
            QMessageBox.warning(self, "Input Error", error_message)
        else:
            weights = [
                float(self.entries[f"Prototype {i} - weight:"].text().strip())
                for i in range(4)
            ]
            wsum = sum(weights)
            if wsum <= 0.0:
                QMessageBox.warning(
                    self,
                    "Input Error",
                    "Prototype weights must have a positive total so they can be normalized.",
                )
                return
            self.result = {
                label: float(entry.text().strip())
                for label, entry in self.entries.items()
            }
            super().accept()


class EventGenerator(_WidgetBase):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("911 Call Data Generator")
        # Input dialog for type_ratio and prototypes
        # These will allow the user to input custom type ratios and prototypes via separate dialogs
        self.type_ratio_dialog = None
        self.prototypes_dialog = None

        # Dictionary for holding type_ratio and prototypes
        self.type_ratios = {}  # Example: {'Law': 0.64, 'EMS': 0.18, 'Fire': 0.18}
        self.prototypes = {}  # Example: {0: {'mu_r': 0.0005, 'sdev_r': 0.0001}, ...}
        self.prototype_weights = {}

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Input field for selecting the graph file (.graphml)
        graph_file_label = QLabel(GRAPH_FILE_FIELD)
        self.graph_file_label = QLineEdit()
        layout.addWidget(graph_file_label)
        layout.addWidget(self.graph_file_label)

        # Button to browse and select a graph file
        graph_file_button = QPushButton("Browse")
        graph_file_button.clicked.connect(self.browse_file)
        layout.addWidget(graph_file_button)

        # Input labels and fields for various parameters
        # These fields collect user inputs for parameters like event timing, duration, etc.
        self.labels = [
            "Graph ID:",  # Insert graph labels
            "First (seconds):",  #  The time of the first event or call in the dataset, measured in seconds from a reference point (e.g., the start of the logging period).
            "Last (seconds):",  # The time of the last event or call in the dataset, measured in seconds from the same reference point.
            "Mean Time Interval (seconds):",  # The average time interval between consecutive 911 calls, measured in seconds.
            "Dead Time after Event (seconds):",  # The average time period after an event during which no new events or calls are expected to occur, measured in seconds. This could represent a cooldown period or a time when the system is not actively logging new calls.
            "Mean Call Interval after incident (seconds):",  #  The average time interval between the end of an incident and the next 911 call, measured in seconds. This could be used to model the frequency of follow-up calls or related incidents.
            "Mean Duration (seconds):",  # The average duration of a 911 call or incident, measured in seconds. This includes the time from the start of the call to its conclusion.
            "Minimum Duration (seconds):",  # The shortest duration of a 911 call or incident in the dataset, measured in seconds. This could be used to filter out very short or incomplete calls.
            "Mean Patience Time (seconds):",  # The average time a caller is willing to wait on hold before hanging up, measured in seconds. This metric is important for understanding caller behavior and optimizing call center operations.
            "Mean On-Site Time (seconds):",  # The average time emergency responders spend on-site at an incident, measured in seconds. This includes the time from arrival at the scene to departure.
            SEED_FIELD_LABEL,  # Integer seed for numpy.random; leave blank to keep the existing RNG state unchanged.
        ]

        self.entries = {}
        for label_text in self.labels:
            label = QLabel(label_text)
            entry = QLineEdit()
            self.entries[label_text] = entry

            layout.addWidget(label)
            layout.addWidget(entry)

        # Button to open the "Set Type Ratios" dialog
        set_type_ratio_button = QPushButton("Set Type Ratios")
        set_type_ratio_button.clicked.connect(self.show_type_ratio_dialog)
        layout.addWidget(set_type_ratio_button)

        # Button to open the "Set Prototypes" dialog
        set_prototypes_button = QPushButton("Set Prototypes")
        set_prototypes_button.clicked.connect(self.show_prototypes_dialog)
        layout.addWidget(set_prototypes_button)

        # Button to start the event generation process
        generate_button = QPushButton("Generate Events")
        generate_button.clicked.connect(self.generate_events)
        layout.addWidget(generate_button)

        # Set the layout of the UI
        self.setLayout(layout)
        self.show()

    def show_type_ratio_dialog(self):
        # Opens a dialog to allow the user to set type ratios
        if not self.type_ratio_dialog:
            self.type_ratio_dialog = TypeRatioDialog()

        if self.type_ratio_dialog.exec_() == QDialog.Accepted:
            # Use the entered values from the dialog
            self.type_ratios = {
                label: float(entry.text())
                for label, entry in self.type_ratio_dialog.entries.items()
            }

    def show_prototypes_dialog(self):
        # Opens a dialog to allow the user to set prototypes
        if not self.prototypes_dialog:
            self.prototypes_dialog = PrototypesDialog()

        if self.prototypes_dialog.exec_() == QDialog.Accepted:
            # Extract and reformat prototype values
            prototypes_entries = self.prototypes_dialog.entries.items()
            prototypes_values = {
                label: float(entry.text()) for label, entry in prototypes_entries
            }

            # Organize prototype data into nested dictionaries
            self.prototypes = {}
            self.prototype_weights = {}
            for label, value in prototypes_values.items():
                split_label = label.split(" - ")
                prototype_num = int(split_label[0].split()[-1])
                var_name = split_label[1].rstrip(":")

                if var_name == "weight":
                    self.prototype_weights[prototype_num] = value
                else:
                    if prototype_num not in self.prototypes:
                        self.prototypes[prototype_num] = {}

                    self.prototypes[prototype_num][var_name] = value

    # Function that allows user to browse local files

    def browse_file(self):
        # Allows the user to browse and select a .graphml file
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("GraphML files (*.graphml)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setOptions(options)

        if file_dialog.exec_():
            selected_file = file_dialog.selectedFiles()
            if selected_file[0].endswith(".graphml"):
                self.graph_file_label.setText(selected_file[0])
            else:
                QMessageBox.warning(
                    self, "Invalid File Type", "Please select a .graphml file."
                )

    # Moved existing main methods to take user inputted data
    # Handles invalid inputs (string instead of int, wrong file
    # type but not invalid logic)
    def generate_events(self):
        # Validates inputs and triggers the event generation process
        error_message = ""
        invalid_fields = []

        # Validate user input fields
        for label_text, entry in self.entries.items():
            text = entry.text().strip()
            if not text:
                if label_text == SEED_FIELD_LABEL:
                    continue
                invalid_fields.append(label_text)
            else:
                if label_text == SEED_FIELD_LABEL:
                    try:
                        int(text)
                    except ValueError:
                        invalid_fields.append(label_text)
                else:
                    try:
                        float(text)  # Attempt to convert to float to check validity
                    except ValueError:
                        invalid_fields.append(label_text)

        # Check if type ratios and prototypes are set
        if not self.type_ratios and not self.prototypes:
            error_message += "Please set Type Ratio and Prototype values before generating events."
        elif not self.type_ratios:
            error_message += "Please set Type Ratio values before generating events."
        elif not self.prototypes:
            error_message += "Please set Prototype values before generating events."

        # Validate the graph file input
        graph_file = self.graph_file_label.text().strip()
        if not graph_file:
            invalid_fields.append(GRAPH_FILE_FIELD)
        elif not graph_file.endswith(".graphml"):
            invalid_fields.append(f"{GRAPH_FILE_FIELD} must be a .graphml file.")
        elif not os.path.isfile(graph_file):
            invalid_fields.append(f"{GRAPH_FILE_FIELD} path is not an existing file.")

        # Handle errors and display warnings
        if invalid_fields:
            error_message = "Invalid or empty values in the following fields: (float values only)\n"
            for field in invalid_fields:
                error_message += f"- {field}\n"

        # Proceed with event generation if inputs are valid
        try:
            # Extract user inputs and use them for event generation
            if error_message:
                raise ValueError(error_message)

            # Get other necessary inputs for functions
            first = float(self.entries["First (seconds):"].text())
            last = float(self.entries["Last (seconds):"].text())
            mu = float(self.entries["Mean Time Interval (seconds):"].text())
            pp_dead_t = float(self.entries["Dead Time after Event (seconds):"].text())
            sec_proc_sigma = float(
                self.entries["Mean Call Interval after incident (seconds):"].text()
            )
            duration_mean = float(self.entries["Mean Duration (seconds):"].text())
            duration_min = float(self.entries["Minimum Duration (seconds):"].text())
            patience_mean = float(self.entries["Mean Patience Time (seconds):"].text())
            avg_on_site_time = float(
                self.entries["Mean On-Site Time (seconds):"].text()
            )
            seed_text = self.entries[SEED_FIELD_LABEL].text().strip()
            random_seed = int(seed_text) if seed_text else None
            graph_id = str(self.entries["Graph ID:"].text())

            generate_cluster_point_process_xml(
                graph_file=graph_file,
                graph_id=graph_id,
                first=first,
                last=last,
                mu=mu,
                pp_dead_t=pp_dead_t,
                sec_proc_sigma=sec_proc_sigma,
                duration_mean=duration_mean,
                duration_min=duration_min,
                patience_mean=patience_mean,
                onsite_mean=avg_on_site_time,
                type_ratios=self.type_ratios,
                prototypes=self.prototypes,
                prototype_weights=self.prototype_weights,
                random_seed=random_seed,
            )

            # Display message box indicating completion
            QMessageBox.information(
                self, "Process Complete", "Event generation completed successfully."
            )

        except (ValueError, FileNotFoundError, KeyError) as ve:
            error_box = QMessageBox()
            error_box.setIcon(QMessageBox.Warning)
            error_box.setWindowTitle("Input Error")
            error_box.setText(str(ve))
            error_box.exec_()


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic 911 call XML (cluster point process) for Graphitti."
    )
    parser.add_argument(
        "--config",
        metavar="FILE.json",
        help="Load all tuned parameters from JSON and exit without opening the GUI.",
    )
    parser.add_argument(
        "--write-config",
        metavar="FILE.json",
        nargs="?",
        const="params.json",
        help="Write a boilerplate JSON config to FILE.json (default: params.json) and exit.",
    )
    args = parser.parse_args()

    if args.write_config is not None:
        write_boilerplate_json_config(args.write_config)
        return

    if args.config:
        run_from_json_config(args.config)
        return

    app = QApplication(sys.argv)
    window = EventGenerator()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
