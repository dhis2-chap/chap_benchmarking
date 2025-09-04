#!/bin/bash

# Setup cronjob for check_updates_and_trigger_run.py

# Define the cron job command
CRON_CMD="*/5 * * * * cd /data/chap_benchmarking && source .venv/bin/activate && python check_updates_and_trigger_run.py >> /data/chap_benchmarking/cron.log 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "check_updates_and_trigger_run.py"; then
    echo "Cron job already exists, updating it..."
    # Remove old cron job and add new one
    (crontab -l 2>/dev/null | grep -v "check_updates_and_trigger_run.py"; echo "$CRON_CMD") | crontab -
else
    echo "Adding new cron job..."
    # Add new cron job
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
fi

echo "Cron job has been set up to run every 5 minutes"
echo "Logs will be written to /data/chap_benchmarking/cron.log"