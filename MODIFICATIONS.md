## Modified Files

### deeploglizer/models/base_model.py
- **Date**: 2025-10-07
- **Modified by**: devsus
- **Changes**:
  - added `use_multi_gpu` parameter to `ForcastBasedModel.__init__()` 
  - added multi-GPU detection in `__init__()`
  - added DataParallel wrapper helper method
  - in `fit()` method wrap the model before training
  - enables training across multiple GPUs when `use_multi_gpu=True`
- **License**: Apache 2.0 (original work by LogPAI Team)
