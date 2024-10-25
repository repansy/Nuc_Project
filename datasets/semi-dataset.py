import torch.utils.data as data


class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir, train_filename):
        super(TrainData, self).__init__()
