#!/bin/bash

# Check if pip is installed
if ! command -v pip &> /dev/null
then
    echo "pip could not be found, attempting to install..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
    rm get-pip.py
fi

# Check if npm is installed
if ! command -v npm &> /dev/null
then
    echo "npm could not be found, please install Node.js and npm."
    exit 1
fi

# Check if requirements.txt exists in the current directory
if [ -f "requirements.txt" ]; then
    echo "Installing required packages..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found in the current directory."
    exit 1
fi

echo "All required packages installed successfully."

# Install the required Python packages
echo "Installing required Python packages..."
pip install -r requirements.txt

# Install Truffle globally if it hasn't been installed already
echo "Installing Truffle..."
npm install -g truffle

# Compile and migrate the smart contract
echo "Compiling and migrating the smart contract..."
truffle compile
truffle migrate

# Split and save the dataset in the data folder
echo "Splitting and saving the dataset..."
python model/distribute_mnist_data.py