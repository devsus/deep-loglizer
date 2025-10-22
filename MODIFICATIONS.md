**Modified by**: devsus
- **Changes**:
  - added `use_multi_gpu` parameter to `ForcastBasedModel.__init__()` 
  - added multi-GPU detection in `__init__()`
  - added DataParallel wrapper helper method
  - in `fit()` method wrap the model before training
  - enables training across multiple GPUs when `use_multi_gpu=True`

  - switched from DataParallel to DDP
  - gitignore for dataset files
  
  - edit init. because of pynvml error (miniconda3/../torch/cuda)
