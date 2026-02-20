#!/usr/bin/env bash
# ==========================================================================
# upload_results.sh -- Zip results and upload for download
# Usage: ./upload_results.sh
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS3_ROOT="$SCRIPT_DIR/ns3/ns-3-dev"
WORK_DIR="$NS3_ROOT/scratch/linear-mesh"
SRC_RESULTS="$WORK_DIR/results"
ZIPFILE="/tmp/ccod_results_$(date '+%Y%m%d_%H%M%S').zip"

if [ ! -d "$SRC_RESULTS" ]; then
    echo "ERROR: No results at $SRC_RESULTS"
    exit 1
fi

echo "Zipping results..."
cd "$WORK_DIR"
zip -r "$ZIPFILE" results/
echo "Created: $ZIPFILE ($(du -h "$ZIPFILE" | cut -f1))"

echo ""
echo "Uploading to temp.sh (files expire in 3 days)..."

# Prefer curl if available, else use Python urllib
if command -v curl &>/dev/null; then
    URL=$(curl -s -F "file=@$ZIPFILE" https://temp.sh/upload)
else
    URL=$(python3 -c "
import urllib.request, os
filepath = '$ZIPFILE'
fn = os.path.basename(filepath)
body = b'------Py\r\nContent-Disposition: form-data; name=\"file\"; filename=\"' + fn.encode() + b'\"\r\nContent-Type: application/zip\r\n\r\n'
with open(filepath, 'rb') as f: body += f.read()
body += b'\r\n------Py--\r\n'
req = urllib.request.Request('https://temp.sh/upload', data=body, headers={'Content-Type': 'multipart/form-data; boundary=----Py'})
resp = urllib.request.urlopen(req, timeout=120)
print(resp.read().decode().strip())
")
fi

if [ -n "$URL" ] && echo "$URL" | grep -q "^http"; then
    echo ""
    echo "=========================================="
    echo "DOWNLOAD URL (expires in 3 days):"
    echo "$URL"
    echo "=========================================="
    echo ""
    echo "On local machine:"
    echo "  curl -L '$URL' -o /tmp/ccod_results.zip && unzip -o /tmp/ccod_results.zip -d '$(pwd)/'"
else
    echo "Upload failed. Response: $URL"
    echo "Zip file at: $ZIPFILE"
fi

echo "Zip remains at: $ZIPFILE"
