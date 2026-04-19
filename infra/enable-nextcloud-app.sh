#!/bin/bash
# Run this once after docker-compose up to enable the Smart File Tagger app
echo "Waiting for Nextcloud to be ready..."
sleep 10
docker exec --user www-data smart-file-tagger-nextcloud-1 \
  php occ app:enable smartfiletagger
echo "App enabled. Checking status..."
docker exec --user www-data smart-file-tagger-nextcloud-1 \
  php occ app:list | grep smartfiletagger
