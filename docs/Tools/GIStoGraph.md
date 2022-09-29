# GIS to GEXF Python Script
This Python script converts GIS(Geographic Information Sytem) layer data, into a graph. The graph is printed out in two formats: `.gexf` and `.graphml`. 

The script read in GIS files using GeoPandas, a Python package built on Pandas designed to handle Geographic data. 
It constructs a NetworkX undirected graph by checking the relationships between the different geometries in the GeoPandas data structures.

To represent geographic call data, the script creates a grid of squares that are used get lattitude and longitude information into the simulation by assigning these squares to caller region nodes.

Because this script will be reused for different purposes, efficiency was sacrificed to increase readability.
Python is a language that can be difficult to read and understand, luckily the main dependencies for this script all have excelent documentation which can be found here
   * [GeoPandas](https://geopandas.org/en/stable/docs.html)
   * [Pandas](https://pandas.pydata.org/docs/reference/index.html)
   * [NetworkX](https://networkx.org/documentation/latest/reference/index.html)
## Nodes
   * PSAP(Public Answering Service Points) just contain the Display Name for the PSAP

   * Fire, Ems, and Law nodes all contain the NG911 unique identifier, along with the display name and a representative point that falls somewhere near the middle of the boundary.

   * Caller Region nodes contain the display name of the PSAP they are linked to, along with a list of points representing squares that make up the area covered by the region. There is only one caller region per PSAP.
## File Input/Output
GeoPandas can read several different GIS file formats. `.shp` and `.gpkg` are two of the most common file formats. 

The script is currently looking for five different layers, PSAP Layer, Law Layer, EMS Layer, Fire Layer, and Provisioning Layer. These layers should be placed in the `GIS_data/Layers/` directory in the GIStoGEXF directory. 
If there are different requirements in the future, the parameters can be adjusted in the `gpd.read_file()` functions in the script. 

The script outputs two files to the `graph_files` directory, one with a `.gexf` extension and one with a `.graphml` extension; both files contain the same graph representation but in a different format. The name of these files can be changed by changing the `out_file_name` variable in the script.

More information on the GEXF file standard can be found [here](https://gexf.net/primer.html)
## Installing GeoPandas
GeoPandas is the main dependency for this project, as all of the GIS data is stored in GeoPandas data structures. 
It has several dependencies itself, and the installation guide can be found [here](https://geopandas.org/en/stable/getting_started/install.html#dependencies)
There are several ways to install GeoPandas and it's dependencies, but the easiest method seems to be using the conda package manager found in the Anaconda Distribution of Python.

I recommend downloading Anaconda [here](https://www.anaconda.com/products/distribution). Then using the Anaconda Navigator, run the terminal application and enter 
   `conda install --channel conda-forge geopandas`. 
This will install a suite of packages that includes all of the dependencies for this script. 

There are other methods outlined in the GeoPandas Installation guide, but the method described above was the easiest and most straightforward.
## Massaging Data
This script is currently configured to read in GIS data provided by Washington State, and create a graph representing only King County. 

The data we received from the state was formatted for ArcGIS, which is a paid commercial GIS software that saves GIS information in a way that cannot be read by GeoPandas. 
In order to get the data into a readable form, I had to drop the `.gdb` file from the state into a free GIS software called QGIS by opening a new project and dragging and dropping the data in.
Once the data was in QGIS, I exported the relevant layers as `.gpkg` files, making sure to select the "Replace all selected raw field values by displayed values" box to avoid formatting issues.

When the `.gpkg` files were read into the script, the scripts removes everything outside of King County using the Provisioning Layer. 
The data provided by the state had the PSAPs split up into multiple boundaries, so the script merges the PSAPs that share display names into larger PSAPS.
