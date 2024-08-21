import numpy as np
import math
import lxml.etree as et
import pandas as pd
import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QFileDialog, QMessageBox, QDialog, QDialogButtonBox, QGridLayout
from cluster_point_process_functions import primprocess, add_types, secprocess, add_vertex_events

# Class used for setting type_ratios in the GUI for the cluster point process
class TypeRatioDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Set Type Ratios")
        self.init_ui()
        self.initial_values = {}  # Store initial values

    def init_ui(self):
        layout = QVBoxLayout()

        self.labels = ["Law", "EMS", "Fire"]
        self.entries = {}
        for label_text in self.labels:
            label = QLabel(label_text)
            entry = QLineEdit()
            self.entries[label_text] = entry

            layout.addWidget(label)
            layout.addWidget(entry)

        button_box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.on_cancel_clicked)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self.backup_initial_values()  # Save initial values

    def accept(self):
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
            error_message = "Invalid or empty values in the following fields:\n"
            for field in invalid_fields:
                error_message += f"- {field}\n"
            QMessageBox.warning(self, "Input Error", error_message)
        else:
            self.result = {label: float(entry.text().strip()) for label, entry in self.entries.items()}
            super().accept()

    def backup_initial_values(self):
        self.initial_values = {label: entry.text() for label, entry in self.entries.items()}

    def on_cancel_clicked(self):
        for label, entry in self.entries.items():
            current_text = entry.text().strip()
            if not current_text:  # If the field is empty, reset to initial value
                if label in self.initial_values:
                    entry.setText(self.initial_values[label])

        self.reject()  # Close the dialog

# Class used for setting prototype values in the GUI for the cluster point process
class PrototypesDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Set Prototypes")
        self.init_ui()

    def init_ui(self):
        layout = QGridLayout()
        self.entries = {}

        labels = ["mu_r:", "sdev_r:", "mu_intensity:", "sdev_intensity:"]
        
        for i in range(4):
            prototype_label = QLabel(f"Prototype {i}:")
            layout.addWidget(prototype_label, i, 0)

            for j, label in enumerate(labels):
                entry = QLineEdit()
                self.entries[f"Prototype {i} - {label}"] = entry
                layout.addWidget(QLabel(label), i, j * 2 + 1)
                layout.addWidget(entry, i, j * 2 + 2)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box, 4, 0, 1, 5)

        self.setLayout(layout)

    def accept(self):
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
            error_message = "Invalid or empty values in the following fields:\n"
            for field in invalid_fields:
                error_message += f"- {field}\n"
            QMessageBox.warning(self, "Input Error", error_message)
        else:
            self.result = {label: float(entry.text().strip()) for label, entry in self.entries.items()}
            super().accept()

class EventGenerator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("911 Call Data Generator")
        # Input dialog for type_ratio and prototypes
        self.type_ratio_dialog = None
        self.prototypes_dialog = None
        
        # Dictionary for holding type_ratio and prototypes
        self.type_ratios = {}
        self.prototypes = {}

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Graph File input field
        graph_file_label = QLabel("Select Graph File (.graphml):")
        self.graph_file_label = QLineEdit()
        layout.addWidget(graph_file_label)
        layout.addWidget(self.graph_file_label)

        graph_file_button = QPushButton("Browse")
        graph_file_button.clicked.connect(self.browse_file)
        layout.addWidget(graph_file_button)

        # Input labels for GUI
        self.labels = [
            "Graph ID:",
            "First (seconds):",
            "Last (seconds):",
            "Mean Time Interval (seconds):",
            "Dead Time after Event (seconds):",
            "Mean Call Interval after incident (seconds):",
            "Mean Duration (seconds):",
            "Minimum Duration (seconds):",
            "Mean Patience Time (seconds):",
            "Mean On-Site Time (seconds):",
        ]

        self.entries = {}
        for label_text in self.labels:
            label = QLabel(label_text)
            entry = QLineEdit()
            self.entries[label_text] = entry

            layout.addWidget(label)
            layout.addWidget(entry)

        # Buttons for opening dialogs (Type Ratio, Prototype)
        set_type_ratio_button = QPushButton("Set Type Ratios")
        set_type_ratio_button.clicked.connect(self.show_type_ratio_dialog)
        layout.addWidget(set_type_ratio_button)

        set_prototypes_button = QPushButton("Set Prototypes")
        set_prototypes_button.clicked.connect(self.show_prototypes_dialog)
        layout.addWidget(set_prototypes_button)

        # Submit button
        generate_button = QPushButton("Generate Events")
        generate_button.clicked.connect(self.generate_events)
        layout.addWidget(generate_button)
        
        self.setLayout(layout)
        self.show()

    def show_type_ratio_dialog(self):
        if not self.type_ratio_dialog:
            self.type_ratio_dialog = TypeRatioDialog()

        if self.type_ratio_dialog.exec_() == QDialog.Accepted:
            # Use the entered values from the dialog
            self.type_ratios = {label: float(entry.text()) for label, entry in self.type_ratio_dialog.entries.items()}

    def show_prototypes_dialog(self):
        if not self.prototypes_dialog:
            self.prototypes_dialog = PrototypesDialog()

        if self.prototypes_dialog.exec_() == QDialog.Accepted:
            prototypes_entries = self.prototypes_dialog.entries.items()
            prototypes_values = {label: float(entry.text()) for label, entry in prototypes_entries}

            # Reformatting the prototypes dictionary
            self.prototypes = {}
            for label, value in prototypes_values.items():
                split_label = label.split(' - ')
                prototype_num = int(split_label[0].split()[-1])
                var_name = split_label[1].rstrip(':')

                if prototype_num not in self.prototypes:
                    self.prototypes[prototype_num] = {}

                self.prototypes[prototype_num][var_name] = value

    # Function that allows user to browse local files
    def browse_file(self):
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
        error_message = ""
        invalid_fields = []
        
        for label_text, entry in self.entries.items():
            if label_text != "Select Region Grid (.graphml):":
                text = entry.text().strip()
                if not text:
                    invalid_fields.append(label_text)
                else:
                    try:
                        float(text)  # Attempt to convert to float to check validity
                    except ValueError:
                        invalid_fields.append(label_text)

        if not self.type_ratios and not self.prototypes:
            error_message += "Please set Type Ratio and Prototype values before generating events."
        elif not self.type_ratios:
            error_message += "Please set Type Ratio values before generating events."
        elif not self.prototypes:
            error_message += "Please set Prototype values before generating events."
        
        # Error handling
        graph_file = self.graph_file_label.text().strip()
        if not graph_file:
            invalid_fields.append("Select Region Grid (.graphml):")
        elif not graph_file.endswith(".graphml"):
            invalid_fields.append("Select Region Grid (.graphml) must be a .graphml file.")

        if invalid_fields:
            error_message = "Invalid or empty values in the following fields: (float values only)\n"
            for field in invalid_fields:
                error_message += f"- {field}\n"

        # Event handling
        try:
            if error_message:
                raise ValueError(error_message)

            # Get other necessary inputs for functions
            first = float(self.entries["First (seconds):"].text())
            last = float(self.entries["Last (seconds):"].text())
            mu = float(self.entries["Mean Time Interval (seconds):"].text())
            pp_dead_t = float(self.entries["Dead Time after Event (seconds):"].text())
            sec_proc_sigma = float(self.entries["Mean Call Interval after incident (seconds):"].text())
            duration_mean = float(self.entries["Mean Duration (seconds):"].text())
            duration_min = float(self.entries["Minimum Duration (seconds):"].text())
            patience_mean = float(self.entries["Mean Patience Time (seconds):"].text())
            avg_on_site_time = float(self.entries["Mean On-Site Time (seconds):"].text())
                
            # Integration of the event generation code
            if graph_file:
                ###########################################################################
                # PRIMARY EVENTS
                ###########################################################################
                # Start your event generation process here based on the valid inputs
                graph = nx.read_graphml('../../gis2graph/graph_files/spd.graphml')
                graph_id = str(self.entries["Graph ID:"].text())
                graph_attribute = graph.nodes[graph_id]['segments']
                graph_grid = np.array(eval(graph_attribute))
                
                # Seed numpy random number to get consistent results
                np.random.seed(20)
                
                # Call primprocess using the inputs from the interface
                incidents = primprocess(first, last, mu, pp_dead_t, graph_grid)
                print(f'Number of Primary events: {incidents.shape[0]}')

                # Ratios based on NORCOM 2022 report. NORCOM doesn't make a distinction
                # between EMS and Fire call types, so I split it in half.
                # type_ratios = {'Law': 0.64,
                #             'EMS': 0.18,
                #             'Fire': 0.18}
                
                # Generate the incident types based on the type_ratios
                incidents_with_types = add_types(incidents, self.type_ratios)
                
                ###########################################################################
                # SECONDARY EVENTS
                ###########################################################################
                # Define prototypes for location of secondary spatio-temporal points
                # 0.001° is aproximately 111 meters (one footbal field plus both endzones)
                # intensity represent the expected number of points per square unit.
                # TODO: The values used for the prototypes are ballpark values not based on
                #       real data. Althoug, they give us around 70,000 - 75,000 calls in a month,
                #       which is close to what Seattle PD receives with 900,000 calls per year.
                # prototypes = {0: {'mu_r':0.0005, 'sdev_r':0.0001, 'mu_intensity':500000, 'sdev_intensity': 50000},
                #         1: {'mu_r':0.001, 'sdev_r':0.0001, 'mu_intensity':1000000, 'sdev_intensity': 60000},
                #         2: {'mu_r':0.0015, 'sdev_r':0.001, 'mu_intensity':1100000, 'sdev_intensity': 70000},
                #         3: {'mu_r':0.003, 'sdev_r':0.001, 'mu_intensity':1500000, 'sdev_intensity': 60000}}
                
                # Time the secondary process generation
                start_t = time.time()

                print('Generating Secondary events...')

                sec_events = secprocess(sec_proc_sigma, duration_mean, duration_min, patience_mean,
                                        avg_on_site_time, self.prototypes, incidents_with_types)
                
                end_t = time.time()

                print('Elapsed time:', round(end_t - start_t, 4), 'seconds')
                print('Number of Primary Events:', len(incidents_with_types))
                print('Number of Secondary Events:', sec_events.shape[0])

                # Output filenames are generic, will match the filename you inputted
                graph_file_path = self.graph_file_label.text()
                output_file_basename = os.path.basename(graph_file_path)
                output_file_name = os.path.splitext(output_file_basename)[0].upper()
                output_file = output_file_name + "_cluster_point_process.xml"
                # Commented out code that saves to a .csv file
                # sec_events_df = pd.DataFrame(sec_events, columns=['time', 'duration', 'x', 'y', 'type'])
                # sec_events_df.to_csv(output_file, index=False, header=True)

                ###########################################################################
                # TURN CALL LIST INTO AN XML TREE AND SAVE TO FILE
                ###########################################################################
                # The root element
                inputs = et.Element('simulator_inputs')

                output_description_name = output_file_name + "Calls "
                # The data element will contain all calls grouped per vertex
                # Use the filename to dynamically update the description attribute
                data = et.SubElement(inputs, 'data', {"description": f"{output_file_name} Calls - Cluster Point Process", 
                                                    "clock_tick_size": "1",
                                                    "clock_tick_unit": "sec"})

                # Create the vertex element with all its associated calls (events)
                vertex_name = graph.nodes[graph_id]['name']
                data = add_vertex_events(data, graph_id, vertex_name, sec_events)

                tree = et.ElementTree(inputs)
                tree_out = tree.write(output_file,
                                    xml_declaration=True,
                                    encoding='UTF-8',
                                    pretty_print=True)

                print('Secondary process was saved to:', output_file)
                
                # Display message box indicating completion
                QMessageBox.information(
                    self, "Process Complete", "Event generation completed successfully."
                )
            else:
                QMessageBox.warning(
                    self, "Missing File", "Please select a graph file."
                )

        except ValueError as ve:
            error_box = QMessageBox()
            error_box.setIcon(QMessageBox.Warning)
            error_box.setWindowTitle("Input Error")
            error_box.setText(str(ve))
            error_box.exec_()

def main():
    app = QApplication(sys.argv)
    window = EventGenerator()
    sys.exit(app.exec_())

if __name__ == '__main__':
    import pandas as pd
    import networkx as nx
    import time
    main()
