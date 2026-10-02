#!/bin/bash
# Install or update the cron job that runs check_updates_and_trigger_run.py every 15
# minutes from this checkout. CHAP_URL and CHAP_API_TOKEN are read from .env if present.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

CRON_CMD="*/15 * * * * cd $DIR && set -a && [ -f .env ] && . ./.env; set +a; $DIR/.venv/bin/python check_updates_and_trigger_run.py >> $DIR/cron.log 2>&1"

if crontab -l 2>/dev/null | grep -q "check_updates_and_trigger_run.py"; then
    echo "Cron job already exists, updating it..."
else
    echo "Adding new cron job..."
fi
(crontab -l 2>/dev/null | grep -v "check_updates_and_trigger_run.py"; echo "$CRON_CMD") | crontab -

echo "Cron job has been set up to run every 15 minutes"
echo "Logs will be written to $DIR/cron.log"
