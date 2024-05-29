# GraphML Generation

Recently we changed our configuration structure through introducing GraphML as the file that holds the details regarding the initial structure of Neual simulations. As the initial structure of the simulations get larger and more complex, there came a need to have a tool to generate the GraphML files. This tool is a simple python script that uses [networkx](https://networkx.org/documentation/latest/tutorial.html). The script helps define the vertices of the graph and 3 of their attributes: the (x, y) location, the type of neuron (excitatory or inhibitory), and the activeness (whether endogenously active or not) of the neuron.

# Script Details

Currently, the neurons in the simulation are placed in a grid format, meaning that there are `n x n` neurons in a square grid, with each neuron's (x, y) location being a whole number. Therefore, the script defines a `height` variable to determine the size of the grid and the number of neurons.

Afterwards, there are two lists: `ActiveNList` and `InhibitoryNList`. `ActiveNList` defines which neurons are endogenously active (uses a boolean), while `InhibitoryNList` defines which neurons are inhibitory (uses a string); any neuron that isn't in `InhibitoryNList` is defined as excitatory.

To generate the GraphML file, first open the python script and edit the variable and two lists mentioned above. After that, edit the string `example.graphml` to your desired file name (leave the `.graphml` part). Finally, run the python file in the command line to generate the file.

Note: You might have noticed that the `node[active]` attributed is set to either 0 or 1 instead of `True` and `False`. This is because the GraphML file stores them as python booleans, which the c++ code can't parse.

# Important Note Regarding `attr.type`

When you create the file, you may see the keys you have created with a different data type than what you intended.

![image](https://github.com/UWB-Biocomputing/Graphitti/assets/125625083/fa01c56d-c4b4-4674-9837-dd9ed9912e56)

This could be a problem as there will be a data type mismatch, which will break the simulation. You will need to edit the GraphML file so that it the attribute data type is correct.

![image](https://github.com/UWB-Biocomputing/Graphitti/assets/125625083/99a85c87-0cbe-4024-bf8a-d5cc4fbd0640)

