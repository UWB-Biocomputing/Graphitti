import numpy as np
import math
import lxml.etree as et

def primprocess(first_inp, last_inp, pp_mu, pp_dead_t, region_grid):
    # Generates a set of primary spatio-temporal events between first_inp and last_inp.
    # The time intervals between events are exponentially distributed with
    # mean pp_mu. The events are then uniformly distributed between the segments
    # of the region_grid, which serve as a constrain box for randomly selecting
    # the (x, y) location.
    # The pp_dead_t is the dead time after an event. It helps to avoid having 2
    # events ocurring at the exact same time. Finally, the times are given in seconds.
    events = np.array([first_inp])
    aveInt = pp_mu + pp_dead_t

    # Generate all the primary processes between first and lastInp
    # drawing the interval between event from an exponential
    # distribution
    while events[-1] < last_inp:
        numInts = int(np.round((last_inp - events[-1]) / aveInt)) + 1
        newInts = np.random.exponential(scale=pp_mu, size=numInts) + pp_dead_t
        newInts = np.cumsum(newInts)
        events = np.concatenate([events, newInts + events[-1]])

    # Include only events between first and lastInp
    if events[-1] > last_inp:
        events = events[events <= last_inp]

    # Add spatial dimension to the primary process.
    # Create a numpy array of uniformly distributed random segments drawn
    # from the region_grid
    n = len(events)
    rand_segments = region_grid[np.random.randint(0, len(region_grid), n)]
    # Generate x and y from the 2 corners defined in each segment
    x = np.random.uniform(rand_segments[:,0,0], rand_segments[:,1,0])
    y = np.random.uniform(rand_segments[:,0,1], rand_segments[:,1,1])

    return np.column_stack((np.round(events).astype(np.int64), x, y))


def add_types(events, type_ratios):
    # We will uniformily distribute the events between 3 types: EMS, Fire, and Law.
    # Assigned a number from 0 to 99 using a uniform distribution, which will be
    # used as a threshold for randomly selecting the the emergency type based
    # on their type_ratios.
    random_ratio = np.random.randint(0, 100, events.shape[0])
    
    # With the ratios sorted in ascending order we set thresholds for the various
    # incident types using their cummulative sum. For instance, if we have ratios of
    # 15% EMS, 15% Fire, and 70% Law; we assign EMS when the uniformly distributed
    # random variable gets a value less than 15, Fire is assigned when it gets a
    # value between 15 and 29, and Law for values between 30 and 99.
    prev_threshold = 0
    cond_list = []
    choice_list = []
    for key, value in sorted(type_ratios.items(), key=lambda x:x[1]):
        threshold = prev_threshold + value * 100
        cond_list.append(random_ratio < threshold)
        # print('Threshold:', key, '= ', threshold)
        choice_list.append(key)
        prev_threshold = threshold
        
    type_list = np.select(cond_list, choice_list)
    return np.column_stack((events, type_list.astype('object')))


def secprocess(sp_sigma, duration_mean, duration_min, prototypes, prim_evts):
    # Secondary process for clustering. Selects a prototype
    # from the dictionary of prototypes, which is used as the magnitude
    # an spread of the primary event. This determines the number of 
    # secondary events generated and their distribution in space.
    # It then generates a secondary point process by attaching
    # to each event in prim_evts a new cluster, with the interval between the
    # primary event and each secondary event being taken from an exponential
    # distribution with mean sp_sigma.
    # 
    # Each secondary events gets a duration drawn from an exponential
    # distribution with mean duration_mean, and a location (x, y) according
    # to the selected prototype.
    #
    # It returns the secondary events containing:
    # [time, duration, x, y, type].

    # Constraints:
    # 1. Values drawn from an exponential distribution get their outliers removed.
    #    The outliers are determines using Tukey's Fence criteria for the upper fence,
    #    calculated as (ln(4) + 1.5 * ln(3)) * SPSigma

    # The prototypes are selected base of 4 classes (0-3) where:
    #   class 0 = 40% of events
    #   class 1 = 50% of events
    #   class 2 = 9% of events
    #   class 3 = 1% of events
    # Each of this classes has a mean and standard deviation for the radius
    # and intensity of the generated secondary process
    proto_class = np.random.rand(len(prim_evts))
    proto_class[(proto_class >= 0.99)] = 3
    proto_class[proto_class < 0.4] = 0
    proto_class[(proto_class >= 0.4) & (proto_class < 0.9)] = 1
    proto_class[(proto_class >= 0.9) & (proto_class < 0.99)] = 2

    sec_evts_t = np.zeros(0) #np.zeros(len(primEvts) * expected_points_num)
    sec_evts_x = np.zeros(0) #np.zeros(len(primEvts) * expected_points_num)
    sec_evts_y = np.zeros(0) #np.zeros(len(primEvts) * expected_points_num)
    sec_evts_cid = np.zeros(0)

    # We need to compute the actual clusters on a per primary event basis
    for pe_num in range(len(prim_evts)):
        # get the radius and intensity
        pcls = proto_class[pe_num]
        # print('protoclass:', pcls)
        radius = np.random.normal(prototypes[pcls]['mu_r'],
                                  prototypes[pcls]['sdev_r'],
                                  size=1)[0]
        intensity = np.random.normal(prototypes[pcls]['mu_intensity'],
                                     prototypes[pcls]['sdev_intensity'],
                                     size=1)[0]
        
        expected_points_num = int(intensity * np.pi * radius**2)
        # Ensure that at least 1 call is generated
        if expected_points_num == 0:
            expected_points_num = 1
        
        # Use Tukey's boxplot method for calculating the fence for the outliers. Based on
        # Sim et al. (2005), the lower fence for an exponential distribution is effectively 0.
        # therefore, we only need to calculate the upper fence. The upper fence is calculated
        # as 1.5 times the interquartile range (IQR) above the third quartile (Q3), for an
        # exponential distribution those values are estimated as follows:
        #   UF = Q3 + 1.5 * IQR
        #   lambda = 1/scale_parameter
        #   Q3 = ln(4)/lambda = ln(4) * scale_parameter
        #   IQR = ln(3)/lambda = ln(3) * scale_parameter
        upper_fence = (math.log(4) + 1.5 * math.log(3)) * sp_sigma

        # Generate the clusters
        actClust = np.random.exponential(scale=sp_sigma, size=expected_points_num)
        outliers = np.where(actClust > upper_fence)[0]
        while len(outliers) > 0:
            actClust[outliers] = np.random.exponential(scale=sp_sigma, size=len(outliers))
            outliers = np.where(actClust > upper_fence)[0]
        
        sec_evts_t_tmp = prim_evts[pe_num][0] + actClust.reshape(expected_points_num)
        sec_evts_t = np.append(sec_evts_t, sec_evts_t_tmp)

        # We will locate the secondary events within a circle, with the primary event at the center
        center_x = prim_evts[pe_num][1]
        center_y = prim_evts[pe_num][2]

        # Generate polar coordinates in the circle
        r = np.random.uniform(0, radius, size=expected_points_num)
        theta = np.random.uniform(0, 2*np.pi, size=expected_points_num)

        # convert polar to Cartesian coordinates
        sec_evts_x_tmp = center_x + r * np.cos(theta)
        sec_evts_x = np.append(sec_evts_x, sec_evts_x_tmp)
        sec_evts_y_tmp = center_y + r * np.sin(theta)
        sec_evts_y = np.append(sec_evts_y, sec_evts_y_tmp)

        # assign the type of the primary event
        e_type = prim_evts[pe_num][3]
        sec_evts_cid = np.append(sec_evts_cid, np.full(expected_points_num, e_type))
        
    # Sort events by time, keeping x and y in sync
    indices = np.argsort(sec_evts_t)
    sec_evts_t = sec_evts_t[indices]
    sec_evts_x = sec_evts_x[indices]
    sec_evts_y = sec_evts_y[indices]
    sec_evts_cid = sec_evts_cid[indices]

    # Draw call duration from an exponential distribution.
    # We also trim outliers using the Tukey's Fences criteria
    duration_fence = (math.log(4) + 1.5 * math.log(3)) * duration_mean
    sec_evts_duration = np.random.exponential(scale=duration_mean, size=len(sec_evts_t))
    outliers = np.where(sec_evts_duration > duration_fence)[0]
    while len(outliers) > 0:
        sec_evts_duration[outliers] = np.random.exponential(scale=duration_mean, size=len(outliers))
        outliers = np.where(sec_evts_duration > duration_fence)[0]

    # Shift call duration distribution to the right by duration_min seconds to
    # avoid calls with 0 duration
    sec_evts_duration = sec_evts_duration + duration_min

    # Reshape numpy arrays so we can concatenate them column wise
    sec_evts_t = sec_evts_t.reshape(-1, 1)
    sec_evts_x = sec_evts_x.reshape(-1, 1)
    sec_evts_y = sec_evts_y.reshape(-1, 1)
    sec_evts_cid = sec_evts_cid.reshape(-1, 1)
    sec_evts_duration = sec_evts_duration.reshape(-1, 1)

    sec_evts = np.concatenate((np.round(sec_evts_t).astype(np.int64),
                               sec_evts_duration.astype(np.int64),
                               sec_evts_x.astype(np.float64),
                               sec_evts_y.astype(np.float64),
                               sec_evts_cid.astype(object)),
                               axis=1)

    return sec_evts
 

def add_vertex_events(node, vertex_id, vertex_name, data):
    # Adds all rows in data as event elements of a vertex to
    # the given xml element tree node.
    # The vertex node gets the vertex_id as vertex_name as attributes,
    # and each event will have: time, duration, x, y, type, and vertex_id as
    # element attribues.
    vertex = et.SubElement(node, 'vertex', {'id': vertex_id, 'name': vertex_name})
    # This is to make sure we don't have calls happening at the exact same second
    prev_time = -1
    for row in data:
        d = dict()
        d['time'] = row[0]
        d['duration'] = row[1]
        d['x'] = row[2]
        d['y'] = row[3]
        d['type'] = row[4]
        d['vertex_id'] = vertex_id

        # Ensure that we don't have calls at the same second
        if d['time'] <= prev_time:
            d['time'] = prev_time + 1
        prev_time = d['time']

        # convert everything to string
        for k, v in d.items():
            d[k] = str(v)
        
        # Add the event to the vertex
        event = et.SubElement(vertex, 'event', d)

    return node


if __name__ == '__main__':
    import pandas as pd
    import networkx as nx
    import time

    # Get the grid for the Seattle PD Caller Region from the graphml file
    graph_file = '../gis2graph/graph_files/spd.graphml'
    spd_cr_id = '194'
    graph = nx.read_graphml(graph_file)
    spd_grid = np.array(eval(graph.nodes[spd_cr_id]['segments']))

    # Define prototypes for location of secondary spatio-temporal points
    # 0.001° is aproximately 111 meters (one footbal field plus both endzones)
    # intensity represent the expected number of points per square unit.
    # TODO: The values used for the prototypes are ballpark values not based on
    #       real data. Althoug, they give us around 70,000 - 75,000 calls in a month,
    #       which is close to what Seattle PD receives with 900,000 calls per year.
    prototypes = {0: {'mu_r':0.0005, 'sdev_r':0.0001, 'mu_intensity':500000, 'sdev_intensity': 50000},
              1: {'mu_r':0.001, 'sdev_r':0.0001, 'mu_intensity':1000000, 'sdev_intensity': 60000},
              2: {'mu_r':0.0015, 'sdev_r':0.001, 'mu_intensity':1100000, 'sdev_intensity': 70000},
              3: {'mu_r':0.003, 'sdev_r':0.001, 'mu_intensity':1500000, 'sdev_intensity': 60000}}
    
    # TODO: These are ballpark values, we need to find these parameter estimates based on real data.
    # All values are in seconds.
    sec_proc_sigma = 20 # Mean of call interval after incident

    ###########################################################################
    # PRIMARY EVENTS
    ###########################################################################
    # Generate incidents for one day 24 hrs * 36000 = 86400 seconds
    # Seattle PD PSAP dispatches to 318069 incidents a year that is,
    # 36.309 avg incidents per hour or an average of 99 seconds between
    # incidents.
    first = 0
    last = 86400  # one day in seconds
    mu = 99 # seconds between incidents
    # The dead time helps with having too many calls at the exact same time
    pp_dead_t = 10   # (seconds)

    # Ratios based on NORCOM 2022 report. NORCOM doesn't make a distinction
    # between EMS and Fire call types, so I split it in half.
    type_ratios = {'Law': 0.64,
                   'EMS': 0.18,
                   'Fire': 0.18}

    # Seed numpy random number to get consistent results
    np.random.seed(20)

    incidents = primprocess(first, last, mu, pp_dead_t, spd_grid)
    print(f'Number of Primary events: {incidents.shape[0]}')

    # Generate the incident types based on the type_ratios
    incidents_with_types = add_types(incidents, type_ratios)

    ###########################################################################
    # SECONDARY EVENTS
    ###########################################################################
    # Time the secondary process generation
    start_t = time.time()

    print('Generating Secondary events...')
    # From the Seattle PD September 2020 data we obtained the following central point
    # and spread estimates:
    #   mean = 204.72
    #   std dev = 222.57
    # This is consistent with exponentially distributed values where the mean and
    # standard deviation are equal.
    duration_mean = 205
    duration_min = 4 # seconds (based on Seattle PD September 2020 data)
    sec_events = secprocess(sec_proc_sigma, duration_mean, duration_min,
                            prototypes, incidents_with_types)
    
    end_t = time.time()
    print('Elapsed time:', round(end_t - start_t, 4), 'seconds')
    print('Number of Primary Events:', len(incidents_with_types))
    print('Number of Secondary Events:', len(sec_events))

    output_file = 'SPD_cluster_point_process.xml'
    # Commented out code that saves to a .csv file
    # sec_events_df = pd.DataFrame(sec_events, columns=['time', 'duration', 'x', 'y', 'type'])
    # sec_events_df.to_csv(output_file, index=False, header=True)

    ###########################################################################
    # TURN CALL LIST INTO AN XML TREE AND SAVE TO FILE
    ###########################################################################
    # The root element
    inputs = et.Element('simulator_inputs')

    # The data element will contain all calls grouped per vertex
    data = et.SubElement(inputs, 'data', {"description": "SPD Calls - Cluster Point Process", 
                                          "clock_tick_size": "1",
                                          "clock_tick_unit": "sec"})
    
    # Create the vertex element with all its associated calls (events)
    vertex_name = graph.nodes[spd_cr_id]['name']
    data = add_vertex_events(data, spd_cr_id, vertex_name, sec_events)

    tree = et.ElementTree(inputs)
    tree_out = tree.write(output_file,
                          xml_declaration=True,
                          encoding='UTF-8',
                          pretty_print=True)

    print('Secondary process was saved to:', output_file)