#!/bin/bash

# Exit on error
set -e

echo "Starting deployment..."

# Pull latest changes from git
git pull origin main

# Setup or update cronjob
echo "Setting up cronjob..."
./setup_cron.sh

echo "Deployment completed successfully!"
