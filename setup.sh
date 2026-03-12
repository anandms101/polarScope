#!/bin/bash

set -e

python3 -m venv .venv
echo "Virtual environment created at .venv"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "Virtual environment activated."
else
  echo "Could not find activation script. Please activate the virtualenv manually."
fi

pip install --upgrade pip
pip install -r requirements.txt

echo "Dependencies installed. You can now run:"
echo "  source .venv/bin/activate"
echo "  python train.py"
echo "  streamlit run app.py"

