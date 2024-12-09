#!/bin/bash

# Set up the Python environment
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Run the pipeline (main.py is the entry script for the pipeline)
python src/main.py "$@"


chmod +x run.sh



