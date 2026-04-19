#!/bin/bash
# Waits for Nextcloud to fully initialize then enables the app
# Run this after docker-compose up

CONTAINER="smart-file-tagger-nextcloud-1"
MAX_WAIT=120
WAITED=0

echo "Waiting for Nextcloud to initialize..."
while [ $WAITED -lt $MAX_WAIT ]; do
    STATUS=$(curl -s http://localhost:8080/status.php 2>/dev/null)
    if echo "$STATUS" | grep -q '"installed":true'; then
        echo "Nextcloud is ready!"
        break
    fi
    echo "Not ready yet... ($WAITED s)"
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: Nextcloud did not start in time"
    exit 1
fi

# Fix permissions on custom_apps directory
docker exec "$CONTAINER" chown -R www-data:www-data /var/www/html/custom_apps/

# Enable our app
docker exec --user www-data "$CONTAINER" php occ app:enable smartfiletagger

docker exec --user www-data smart-file-tagger-nextcloud-1 \
  php occ config:system:set trusted_domains 1 --value=nextcloud
echo "Trusted domains configured."

# Verify
echo "Checking app status..."
docker exec --user www-data "$CONTAINER" php occ app:list | grep -A1 "smartfiletagger"

echo "Done. Open http://localhost:8080 and log in as admin/admin"
