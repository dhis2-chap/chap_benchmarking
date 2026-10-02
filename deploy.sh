#!/bin/bash
# Deploy this checkout on the benchmarking server: run from the checkout, or by the
# GitHub deploy workflow over SSH. Pulls main, syncs the venv and (re)installs the cron job.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"   # uv, on a non-login shell

echo "Starting deployment..."
git pull origin main
uv sync --no-dev

echo "Setting up cronjob..."
./setup_cron.sh

echo "Deployment completed successfully!"
