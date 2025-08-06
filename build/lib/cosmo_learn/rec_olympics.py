import numpy as np

class OlympicsMaster:
    def __init__(self):
        self.scores = []
        self.scores_dict = {}

    def olympics_master(self, metric, metrics):
        # Initialize scores list with zeros if it's empty
        if not self.scores:
            self.scores = [0] * len(metrics)
            self.scores_dict = {"D0": [0] * len(metrics), "D1": [0] * len(metrics), \
                                "D2": [0] * len(metrics), "DW": [0] * len(metrics), \
                                "Ch2_H0": [0] * len(metrics)}

        if metric == "D0":
            # Method with D0 closest to 1/2 gets +1
            # min_index = metrics.index(min(metrics))
            min_index = min(range(len(metrics)), key=lambda i: abs(metrics[i] - 0.5))
            self.scores[min_index] += 1
            self.scores_dict["D0"][min_index] += 1
        elif metric == "D1":
            # Method with highest D1 gets +1
            max_index = metrics.index(max(metrics))
            self.scores[max_index] += 1
            self.scores_dict["D1"][max_index] += 1
        elif metric == "D2":
            # Method with D2 closest to 1 gets +1
            # min_index = metrics.index(min(metrics))
            min_index = min(range(len(metrics)), key=lambda i: abs(metrics[i] - 1))
            self.scores[min_index] += 1
            self.scores_dict["D2"][min_index] += 1
        elif metric == "DW":
            # Method closest to 2 in absolute value gets +1
            min_index = min(range(len(metrics)), key=lambda i: abs(metrics[i] - 2))
            self.scores[min_index] += 1
            self.scores_dict["DW"][min_index] += 1
        elif metric == "Ch2_H0":
            # Method with Ch2_H0 closest to 1 gets +1
            # min_index = metrics.index(min(metrics))
            min_index = min(range(len(metrics)), key=lambda i: abs(metrics[i] - 1))
            self.scores[min_index] += 1
            self.scores_dict["Ch2_H0"][min_index] += 1
        else:
            print("Invalid metric specified.")

        return self.scores