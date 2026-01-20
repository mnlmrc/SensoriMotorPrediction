

class Dataset3D:

    def __init__(self, measurements, descriptors=None,
                 obs_descriptors=None, channel_descriptors=None):
        if measurements.ndim != 3:
            raise AttributeError(
                "measurements must be in dimension n_obs x n_channel")
        self.measurements = measurements
        self.n_obs, self.n_channel, self.timepoints = self.measurements.shape
        self.descriptors = descriptors
        self.obs_descriptors = obs_descriptors
        self.channel_descriptors = channel_descriptors






