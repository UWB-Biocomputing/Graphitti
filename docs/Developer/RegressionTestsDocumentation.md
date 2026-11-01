# NG911 Tests
This document outlines the parameters used by NG911 regression tests. There are three files required to run an NG911 test; the Configuration file, the Graph file, and the Input calls files. For each of these files, a table is provided showing the main parameters in the file and their values for the existing NG911 tests.

# Configuration files
| Parameter | test-small-911.xml | test-medium-911.xml |
|:------|:------:|:------:|
| Epoch duration | 900 | 200 |
| Number of epochs | 2 | 1440 |
| Redial probability | 0.85 | 0.85 |
| Average driving speed | 30 | 30 |

# Graph files
| Parameter | test-small-911.graphml | test-medium-911.graphml |
|:------|:------:|:------:|
| Number of Vertices | 12 | 1932 |
| Number of Caller Regions | 1 | 21 |
| Number of PSAPs | 1 | 21 |
| Min number of trunks for PSAPs | 5 | 5 |
| Max number of trunks for PSAPs | 5 | 10 |
| Min number of servers for PSAPs | 4 | 3 |
| Max number of servers for PSAPs | 4 | 5 |
| Number of EMS Responders | 3 | 630 |
| Number of Law Responders | 4 | 630 |
| Number of Fire Responders | 2 | 630 |
| Min number of trunks for Responders | 5 | 6 |
| Max number of trunks for Responders | 10 | 12 |
| Min number of servers for Responders | 3 | 3 |
| Max number of servers for Responders | 5 | 6 |

# Input calls files
The parameters for the Input calls table are taken from the cluster_point_process.py file in Graphitti/Tools/InputGeneration/ClusterPointProcess

| Parameter | test-medium-911-calls.xml |
|:------|:------:|
| Number of emergency calls | 34,119 |
| First (seconds) | 34 |
| Last (seconds) | 32436 |
| Mean Time Interval (seconds) | 62.88 |
| Dead Time after Event (seconds) | 1 |
| Mean Call Interval after incident (seconds) | 20 |
| Mean Duration (seconds) | 204 |
| Minimum Duration (seconds) | 4 |
| Mean Patience Time (seconds) | 50 |
| Mean On-Site Time (seconds) | 1200 |
| Type Ratio Law | 0.33 |
| Type Ratio EMS | 0.33 |
| Type Ratio Fire | 0.33 |
| Prototype 0 mu_r | 0.0005 |
| Prototype 0 sdev_r | 0.0001 |
| Prototype 0 mu_intensity | 500000 |
| Prototype 0 sdev_intensity | 50000 |
| Prototype 1 mu_r | 0.001 |
| Prototype 1 sdev_r | 0.0001 |
| Prototype 1 mu_intensity | 1000000 |
| Prototype 1 sdev_intensity | 60000 |
| Prototype 2 mu_r | 0.0015 |
| Prototype 2 sdev_r | 0.001 |
| Prototype 2 mu_intensity | 1100000 |
| Prototype 2 sdev_intensity | 70000 |
| Prototype 3 mu_r | 0.003 |
| Prototype 3 sdev_r | 0.001 |
| Prototype 3 mu_intensity | 1500000 |
| Prototype 3 sdev_intensity | 60000 |
