from off_moo_bench.datasets.continuous_dataset import ContinuousDataset
from off_moo_bench.disk_resource import DiskResource

TENames = [
    "5g",
]

def _get_x_files_from_name(env_name):
    return [f"{env_name}/{env_name}-x-0.npy"]

def _get_x_test_files_from_name(env_name):
    return [f"{env_name}/{env_name}-test-x-0.npy"]

class TOYEXAMPLEDataset(ContinuousDataset):
    
    name = "5g"
    x_name = "input_values"
    y_name = "output_values"
    
    @classmethod
    def register_x_shards(cls):
        return [DiskResource(file, is_absolute=False,)
               for file in _get_x_files_from_name(cls.name)]
    
    @classmethod
    def register_y_shards(cls):
        return [DiskResource(file.replace('-x-', '-y-'), is_absolute=False,)
               for file in _get_x_files_from_name(cls.name)]
        
    @classmethod
    def register_x_test_shards(cls):
        return [DiskResource(file, is_absolute=False,)
               for file in _get_x_test_files_from_name(cls.name)]
    
    @classmethod
    def register_y_test_shards(cls):
        return [DiskResource(file.replace('-x-', '-y-'), is_absolute=False,)
               for file in _get_x_test_files_from_name(cls.name)]
    
    def __init__(self, dataset_max_percentile=1.0, dataset_min_percentile=0.0, **kwargs):
        self.name = self.name.lower()
        # print(f"self.name: {self.name}")
        assert self.name in TENames
        super(TOYEXAMPLEDataset, self).__init__(
            self.register_x_shards(),
            self.register_y_shards(),
            self.register_x_test_shards(),
            self.register_y_test_shards(),
            **kwargs
        )
        self.dataset_max_percentile = dataset_max_percentile
        self.dataset_min_percentile = dataset_min_percentile


class FiveGDataset(TOYEXAMPLEDataset):
    name = "5g"

 

if __name__ == "__main__":
    dataset = FiveGDataset()