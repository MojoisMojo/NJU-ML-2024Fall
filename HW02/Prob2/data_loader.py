class DataLoader:
    def __init__(self, data_path = None):
        self.data_path = data_path

    def load_data(self):
        if self.data_path is None:
            print("Data path is not specified.")
            return None
        with open(self.data_path, 'r') as f:
            data = f.readlines()
        return data

