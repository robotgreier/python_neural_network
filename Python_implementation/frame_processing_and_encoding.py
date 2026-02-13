import cv2 as cv
import numpy as np

class FrameProcessor:
    """
    Class for processing frames - detecting keypoints for each current frame and returning
    """
    def __init__(self, threshold_edges, n_bins_x=4, n_bins_y=3, frame_width=640, frame_height=480):
        self.threshold_edges = threshold_edges
        self.n_bins_x = n_bins_x
        self.n_bins_y = n_bins_y
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Internal state to store the previous frame keypoint
        self.kp_counts_old = None
        self.orb = cv.ORB_create()

    def encoder_keypoints_event_driven(self, new_val, prev_val):
        """
        Returns a spike (1) if the value has moved to a different threshold interval.
        """
        if new_val == prev_val:
            return 0

        # Use digitize to find which interval index the values fall into
        new_interval_idx = np.digitize(new_val, self.threshold_edges)
        prev_interval_idx = np.digitize(prev_val, self.threshold_edges)

        if new_interval_idx != prev_interval_idx:
            return 1
            
        return 0

    def process_and_encode_frame(self, current_frame_path, response_cutoff=0.000):
        """
        Processes a single frame and compares it to the stored previous frame 
        to return a spike train based on threshold intervals.
        """
        kp_counts = []
        spike_train = []

        # Read frame
        current_frame = cv.imread(current_frame_path, cv.IMREAD_GRAYSCALE)
        if current_frame is None:
            return []

        # Detect and compute keypoints
        keypoints = self.orb.detect(current_frame, None)
        keypoints, des = self.orb.compute(current_frame, keypoints)

        # Filter keypoints by response
        kp_list = [kp for kp in keypoints if kp.response >= response_cutoff]

        # Extract keypoint coordinates for the frame
        x_data = [kp.pt[0] for kp in kp_list]
        y_data = [kp.pt[1] for kp in kp_list]

        # Bin points and get counts
        hist, x_edges, y_edges = np.histogram2d(
            x=x_data, y=y_data, 
            bins=[self.n_bins_x, self.n_bins_y], 
            range=[[0, self.frame_width], [0, self.frame_height]]
        )

        # Get count for each bin
        for i in range(len(x_edges) - 1):
            for j in range(len(y_edges) - 1):
                kp_counts.append(float(hist[i, j]))

        # Encode logic
        if self.kp_counts_old is not None:
            # Iterate through the bin counts to generate spikes
            for new_val, prev_val in zip(kp_counts, self.kp_counts_old):
                spike_train.append(self.encoder_keypoints_event_driven(new_val, prev_val))
        else:
            # If no previous frame exists, return a zero spike train
            spike_train = [0] * (self.n_bins_x * self.n_bins_y)

        # Save current counts as old for the next timestep
        self.kp_counts_old = kp_counts

        return spike_train