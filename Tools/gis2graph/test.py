import matplotlib.pyplot as plt
import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Polygon
import math


def main():
    # initialize graph
    G = nx.DiGraph()

    region_string = []  # stores the strings of squares that will be set as caller region attributes

    out_file_name = "King_county_NG911"

    square_counter = 0

    # read in GIS layer data
    psap_layer = gpd.read_file("GIS_data/Layers/PSAP_layer.gpkg")
    ems_layer = gpd.read_file("GIS_data/Layers/EMS_layer.gpkg")
    law_layer = gpd.read_file("GIS_data/Layers/Law_layer.gpkg")
    fire_layer = gpd.read_file("GIS_data/Layers/Fire_layer.gpkg")
    provisioning_layer = gpd.read_file("GIS_data/Layers/Provisioning_layer.gpkg")

    # create series of boolean values denoting whether geometry is within King County
    psap_within_kc = psap_layer.within(provisioning_layer.iloc[21].geometry)
    ems_within_kc = ems_layer.within(provisioning_layer.iloc[21].geometry)
    law_within_kc = law_layer.within(provisioning_layer.iloc[21].geometry)
    fire_within_kc = fire_layer.within(provisioning_layer.iloc[21].geometry)

    # create new GeoDataFrames of just items located within King County using series above
    kc_psap = psap_layer.drop(np.where(psap_within_kc == False)[0])
    kc_ems = ems_layer.drop(np.where(ems_within_kc == False)[0])
    kc_law = law_layer.drop(np.where(law_within_kc == False)[0])
    kc_fire = fire_layer.drop(np.where(fire_within_kc == False)[0])

    # area_multiplier gets multiplied by the smallest psap area to determine size of the squares in the grid
    area_multiplier = 10

    # Some of the data from the state had messed up psap names. This line fixes them so they can be consolidated
    kc_psap.loc[[265, 262, 185], 'DsplayName'] = "King County Sheriff's Office - Marine Patrol"

    names = [] # 2 empty lists for storing names and polygons for merging psaps together
    polys = []
    es_nguid = [] # list for storing the nguids

    # Loops through and finds all unique names, as well as sorting all the polygons that make up those regions
    for n in range(kc_psap.shape[0]):
        if (kc_psap.iloc[n].DsplayName) in names:
            polys[names.index(kc_psap.iloc[n].DsplayName)].append(kc_psap.iloc[n].geometry)
        else:
            names.append(kc_psap.iloc[n].DsplayName)
            polys.append([kc_psap.iloc[n].geometry])
            es_nguid.append(kc_psap.iloc[n].ES_NGUID)

    # Takes the lists of polygons, and merges them into new polygons for the creation of the merged_kc_psap GeoDataFrame
    merged_polys = []
    for m in range(len(polys)):
        merged_polys.append(unary_union(polys[m]))

    # Create a new GeoDataFrame with the unique names and merged geometries
    merged_kc_psap = gpd.GeoDataFrame({'DisplayName': names, 'geometry': merged_polys, 'ES_NGUID': es_nguid}, crs=kc_psap.crs)

    # Find the area of the smallest merged psap, use that to determine the square size
    areas = merged_kc_psap.area
    side_length = math.sqrt(areas.min() * area_multiplier)

    # Creates a grid of squares based on the bounds of merged_kc_psaps, and the side_length
    xmin, ymin, xmax, ymax = merged_kc_psap.total_bounds
    cols = list(np.arange(xmin, xmax + side_length, side_length))
    rows = list(np.arange(ymin, ymax + side_length, side_length))
    squares = []
    for x in cols[:-1]:
        for y in rows[:-1]:
            squares.append(
                Polygon([(x, y), (x + side_length, y), (x + side_length, y + side_length), (x, y + side_length)]))
    grid = gpd.GeoDataFrame({'geometry': squares}, crs=kc_psap.crs)

    # show resulting maps
    # merged_kc_psap.plot()
    # kc_ems.plot()
    kc_psap.plot()
    # kc_law.plot()
    # kc_fire.plot()
    plt.show()
main()
