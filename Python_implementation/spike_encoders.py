# Imports
def encoder_keypoints_event_driven(new_val, prev_val, threshold_edges:tuple):
    """
    Simple encoder function that returns a spike (1) whenever two values are not within the same threshold interval.
    
    :param new_val: New value to be compared
    :param prev_val: Previous value
    :param threshold_edges: Tuple of threshold edges
    :type thresholds: tuple
    """

    # Check if there has been any change in values
    if new_val == prev_val:
        # If no change, no spike
        return 0

    # Iterate through threshold intervals
    for i in range(len(threshold_edges) - 1):
        interval = range(threshold_edges[i], threshold_edges[i+1])

        # Check if new val and prev val are withing the same threshold interval
        if (new_val in interval) ^ (prev_val in interval):
            # Spike if not in the same interval
            return 1
        
    # Else no spike
    return 0