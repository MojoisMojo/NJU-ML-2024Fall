# import os
# from utils import print_and_write
# class Manager:
#     def __init__(self, dir_path, task_name, model_name = "") -> None:
#         self.dir_path = dir_path
#         self.task_path = f"{dir_path}/{task_name}_{model_name}.txt"
#         self.curve_path = f"{dir_path}/{task_name}_{model_name}_roc_curve.png"
#         os.makedirs(dir_path, exist_ok=True)
#         out_file = open(self.task_path, "w")
#         out_file.close()
#         self.out_file = open(self.task_path, "a")
#     def print(self, text):
#         print_and_write(self.out_file, text)
#     def run(self):
#         raise NotImplementedError