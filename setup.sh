#!/bin/bash

# Set the name of the virtual environment directory
VENV_DIR=".venv"

# Dataset filename
DATASET_FILE="project-evergreen-dataset_v1.zip"

# URL of the dataset ZIP file
DATASET_URL="https://project-evergreen-datasets.us-sea-1.linodeobjects.com/$DATASET_FILE"  # Replace with your URL

# Directory to extract the dataset
DATA_DIR="data"

# Check if the virtual environment directory exists
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  python3 -m venv $VENV_DIR
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Ensure the virtual environment is activated
if [ "$VIRTUAL_ENV" != "" ]; then
  echo "Virtual environment activated: $VIRTUAL_ENV"
else
  echo "Failed to activate virtual environment"
  exit 1
fi

# Install the requirements
if [ -f "requirements.txt" ]; then
  echo "Installing requirements from requirements.txt..."
  pip install -r requirements.txt
else
  echo "requirements.txt not found"
  exit 1
fi

# Download and extract the dataset
if [ ! -d "$DATA_DIR/train/images" ]; then
  echo "Downloading dataset from $DATASET_URL..."
  curl -L $DATASET_URL -o $DATASET_FILE

  echo "Extracting dataset..."
  unzip $DATASET_FILE -d $DATA_DIR

  echo "Cleaning up..."
  rm $DATASET_FILE
else
  echo "Dataset already exists in $DATA_DIR"
fi

echo "Setup complete. Virtual environment is ready and dataset is downloaded."
