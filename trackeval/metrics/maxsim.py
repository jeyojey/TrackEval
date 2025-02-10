import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing
from .. import utils


class MaxSim(_BaseMetric):
    """Calculates the maximum similarity between any pair of ground truth and tracker detections per sequence based on the similarity_scores for each dataset.

    """

    @staticmethod
    def get_default_config():
        """Default class config values"""
        default_config = {
            'PRINT_CONFIG': True  # Whether to print the config information on init. Default: False.
        }
        return default_config

    def __init__(self, config=None):
        super().__init__()
        self.float_fields = ['MaxSim']
        self.fields = self.float_fields
        self.summary_fields = self.fields

        # Configuration options:
        self.config = utils.init_config(config, self.get_default_config(), self.get_name())

    @_timing.time
    def eval_sequence(self, data):
        """Calculates MaxSim for one sequence"""
        # Initialise results
        res = {}
        for field in self.fields:
            res[field] = 0

        # Return result quickly if tracker or gt sequence is empty
        if data['num_tracker_dets'] == 0 or data['num_gt_dets'] == 0:
            res['MaxSim'] = 0.0
            return res

        max_sim = 0.0
        # Loop over each timestep's similarity score matrix.
        for similarity in data['similarity_scores']:
            if similarity.size > 0:
                current_max = np.max(similarity)
                if current_max > max_sim:
                    max_sim = current_max

        res['MaxSim'] = max_sim
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Combines metrics across all classes by averaging over the class values.
        If 'ignore_empty_classes' is True, then it only sums over classes with at least one gt or predicted detection.
        """
        res = {}
        values = [v['MaxSim'] for v in all_res.values()]
        res['MaxSim'] = np.mean(values) if values else 0.0
        return res

    def combine_classes_det_averaged(self, all_res):
        """For MaxSim, combining by detection is equivalent to class-averaging."""
        return self.combine_classes_class_averaged(all_res)

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        values = [v['MaxSim'] for v in all_res.values()]
        res['MaxSim'] = np.mean(values) if values else 0.0
        return res
