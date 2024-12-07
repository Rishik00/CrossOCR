#!/bin/bash

# Step: Navigate into the cloned repository directory
# This will allow us to execute commands within the CrossOCR project
cd /content/CrossOCR

# Step: Pull the latest changes from the repository
# This ensures that the local repository is up-to-date with any recent updates made to the GitHub repo
git pull

# Step: Install dependencies by running the setup script
# This will install all required dependencies for the project as specified in setup.py
python setup.py install

# Optional: If you want to use this in a virtual environment, you can include instructions like:
# source /path/to/your/venv/bin/activate  # Uncomment if you use a virtual environment

# After running these steps, the CrossOCR project should be set up and ready to use.
