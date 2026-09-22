#!/bin/bash

# Exit on error
set -e

echo "Starting deployment..."

# Pull latest changes from git
git pull origin main
uv sync

# Setup or update cronjob
echo "Setting up cronjob..."
./setup_cron.sh

echo "Deployment completed successfully!"
