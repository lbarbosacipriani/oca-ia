# Setup script for the INCOR environment
# This script creates a virtual environment and installs required packages.
echo "Criando virtual environment..."
python3 -m venv .incor_env
echo "Virtual environment criado com sucesso."
echo "Activating virtual environment..."
source .incor_env/bin/activate
echo "Virtual environment ativado."
echo "Installing packages..."
pip3 install -r requirements.txt
echo "All packages have been installed." 
python3 main.py 