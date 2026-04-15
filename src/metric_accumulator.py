
class MetricAccumulator:
    """ Accumulates metrics over batches and returns the average. """
    def __init__(self):
        self.metrics = {}
        self.count = 0
        
    def update(self, metric_dict):
        self.count += 1
        for k, v in metric_dict.items():
            if k not in self.metrics: 
                self.metrics[k] = 0.0
            self.metrics[k] += v
        
    def get_averages(self):
        if self.count == 0: return {}
        return {k: v / self.count for k, v in self.metrics.items()}
