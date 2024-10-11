**File Structure**
  - Edit and run _main.py_ indicating the model you want to train for the task (see models.py for models)
  - _data_transforms.py_ lists down the data augmentations applied during training
  - _rejection_loss.py_ is the implementation of the rejection loss and can be used in _Train.py_
  - _Train.py_ is used to train and validate the model
  - To add a layer of explainability use _GradCAM.py_

**Related Published Work**

- ResNet18 demonstrated the best performance for LUS frame-level classification in adults
  https://doi.org/10.1016/j.ultras.2023.106994

- ResNet18_SA along with the rejection loss has been used to classify the intensity projected LUS video data
  https://doi.org/10.1016/j.compbiomed.2023.107885
