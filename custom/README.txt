## File Descriptions

### `__init__.py`
Initializes the custom module, making it easier to import the various components.

### `backbone.py`
Defines the backbone model architecture and loading functions.

### `dataset.py`
Handles dataset loading and preprocessing.

### `head.py`
Defines the head models used for classification or regression tasks (GRU and SVR)

### `helper.py`
Contains helper functions and enumerations.
- Enumerations like `CLIPS_REDUCTION`, `EMBEDDING_REDUCTION`, `MODEL_TYPE`, `SAMPLE_FRAME_STRATEGY`, and `HEAD`.

### `model.py`
Defines the main model class and its methods, including methods for training, testing, and feature extraction

### `neck.py`
Defines the neck model architecture, used to apply the feature reduction (spatial or temporal)

### `scripts.py`
Contains various scripts for training, testing, evaluation and visualization.

### `tools.py`
Utility functions for plotting, saving, and loading data.
- Functions for data visualization, saving/loading data, and generating plots.
