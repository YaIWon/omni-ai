#!/bin/bash

# Setup script for GitHub Evolver Production

echo "Setting up GitHub Evolver Production..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p vault/encrypted
mkdir -p vault/logs
mkdir -p vault/recovery
mkdir -p vault/keys
mkdir -p forks
mkdir -p backups

# Set permissions
chmod 700 vault
chmod 700 vault/*

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "# GitHub Evolver Configuration" > .env
    echo "GITHUB_TOKEN=your_github_token_here" >> .env
    echo "VAULT_PASSWORD=!@3456AAbb" >> .env
    echo "" >> .env
    echo "Warning: Please edit .env file and add your GitHub token"
fi

echo "Setup complete!"
echo "Edit .env file to add your GitHub token"
echo "Run: source venv/bin/activate && python github_evolver_production.py"
