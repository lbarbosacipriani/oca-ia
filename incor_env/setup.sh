# Setup script for the INCOR environment
# This script creates a virtual environment and installs required packages.
echo "Creating virtual environment..."
python3 -m venv .incor_env
echo "Virtual environment created successfully."
echo "Activating virtual environment..."
source .incor_env/bin/activate
echo "Virtual environment activated."
echo "Installing packages..."
pip3 install -r requirements.txt
echo "All packages have been installed." 
python3 main.py 