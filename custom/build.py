from transformers import AutoImageProcessor
from custom.dataset import customDataset

class dataset_builder:
  def __init__(self,path_dataset, path_labels, preprocess, sample_frame_strategy):
    self.path_dataset = path_dataset
    self.preprocess = preprocess
    self.path_labels = path_labels 
    
  def build(self):
    processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    custom_ds = customDataset(path_dataset=self.path_dataset, 
                          path_labels=self.path_labels,
                          preprocess=processor)
