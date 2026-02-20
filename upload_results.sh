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
echo "Uploading to file.io..."
RESPONSE=$(curl -s -F "file=@$ZIPFILE" https://file.io/?expires=1d)
URL=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('link','FAILED'))" 2>/dev/null || echo "")

if [ -n "$URL" ] && [ "$URL" != "FAILED" ]; then
    echo ""
    echo "=========================================="
    echo "DOWNLOAD URL (expires in 24h, one-time):"
    echo "$URL"
    echo "=========================================="
    echo ""
    echo "On local machine run:"
    echo "  curl -L '$URL' -o /tmp/ccod_results.zip && unzip -o /tmp/ccod_results.zip -d '$SCRIPT_DIR/'"
else
    echo "file.io failed. Trying transfer.sh..."
    URL2=$(curl --upload-file "$ZIPFILE" "https://transfer.sh/ccod_results.zip" 2>/dev/null || echo "")
    if [ -n "$URL2" ]; then
        echo ""
        echo "=========================================="
        echo "DOWNLOAD URL (expires in 14 days):"
        echo "$URL2"
        echo "=========================================="
        echo ""
        echo "On local machine run:"
        echo "  curl -L '$URL2' -o /tmp/ccod_results.zip && unzip -o /tmp/ccod_results.zip -d '$SCRIPT_DIR/'"
    else
        echo ""
        echo "Both uploads failed. Manual transfer needed."
        echo "Zip file is at: $ZIPFILE"
        echo "Size: $(du -h "$ZIPFILE" | cut -f1)"
        echo ""
        echo "Option: Start a simple HTTP server on the remote:"
        echo "  cd /tmp && python3 -m http.server 9999"
        echo "Then on local (replace REMOTE_IP):"
        echo "  curl http://REMOTE_IP:9999/$(basename $ZIPFILE) -o /tmp/ccod_results.zip"
    fi
fi

rm -f "$ZIPFILE"
