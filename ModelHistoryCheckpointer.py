import time
import numpy as np
import keras


class ModelHistoryCheckpointer:
    
    def __init__(self, model, save_dir="./models/"):
        self.save_dir = save_dir
        self.best_vals = [1000000] * (len(model.metrics))
        self.file_index_log = {}
        self.datetime_prefix = time.strftime("%m-%d_%H-%M", time.localtime())
        self.metric_names = [m.__name__ if callable(m) else m for m in model.metrics]
    
    
    def save_on_epoch(self, model, epoch, stats_all, batches_count):
        metric_stats = stats_all[1:]
        for i in range(len(metric_stats)):
            if metric_stats[i] < self.best_vals[i]:
                filename = self.save_dir + "model_" + self.datetime_prefix + "_" + self.metric_names[i] + ".h5"
                model.save(filename)
                self.file_index_log[self.metric_names[i]] = (filename, epoch, stats_all, batches_count)
                self.best_vals[i] = metric_stats[i]
    
    
    def save_last(self, model, epoch, stats_all, batches_count):
        filename = self.save_dir + "model_" + self.datetime_prefix + "_last.h5"
        self.last = model.save(filename)
        self.file_index_log["last"] = (filename, epoch, stats_all, batches_count)



#