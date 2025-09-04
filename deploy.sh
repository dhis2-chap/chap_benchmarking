#!/bin/bash

# Exit on error
set -e

echo "Starting deployment..."

# Pull latest changes from git
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt

# Setup or update cronjob
echo "Setting up cronjob..."
./setup_cron.sh

echo "Deployment completed successfully!"
