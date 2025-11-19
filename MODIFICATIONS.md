**Modified by**: devsus
- **Changes**:
  - added `use_multi_gpu` parameter to `ForcastBasedModel.__init__()` 
  - added multi-GPU detection in `__init__()`
  - added DataParallel wrapper helper method
  - in `fit()` method wrap the model before training
  - enables training across multiple GPUs when `use_multi_gpu=True`

  - switched from DataParallel to DDP
  - OMP number of threads spec
  
  - gitignore for dataset files
  
  - edit init. because of pynvml error (miniconda3/../torch/cuda)
  
  - optimizer Adam outside of training loop
  
  - use `weights_only=True` when loading model’s pure state dict (secure)
  
  - DDP for all models implemented
  
  - use atch_first=true'in Transformer instead of transpose (performance)
  
  - calculate true top-k acc in base_model.py
  
  - fixed 'kwargs' bug in models
  
  - per rank batch scaling when using DDP
