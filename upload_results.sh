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
echo "Uploading via Python..."
python3 -c "
import urllib.request, json, sys, os

filepath = '$ZIPFILE'
filename = os.path.basename(filepath)

# Try file.io
try:
    import subprocess
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    body = b''
    body += ('--' + boundary + '\r\n').encode()
    body += ('Content-Disposition: form-data; name=\"file\"; filename=\"' + filename + '\"\r\n').encode()
    body += b'Content-Type: application/zip\r\n\r\n'
    with open(filepath, 'rb') as f:
        body += f.read()
    body += ('\r\n--' + boundary + '--\r\n').encode()

    req = urllib.request.Request(
        'https://file.io/?expires=1d',
        data=body,
        headers={'Content-Type': 'multipart/form-data; boundary=' + boundary}
    )
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read())
    url = data.get('link', '')
    if url:
        print(f'\n==========================================')
        print(f'DOWNLOAD URL (expires 24h, single use):')
        print(url)
        print(f'==========================================')
        sys.exit(0)
except Exception as e:
    print(f'file.io failed: {e}')

# Try 0x0.st
try:
    boundary = '----Boundary0x0'
    body = b''
    body += ('--' + boundary + '\r\n').encode()
    body += ('Content-Disposition: form-data; name=\"file\"; filename=\"' + filename + '\"\r\n').encode()
    body += b'Content-Type: application/zip\r\n\r\n'
    with open(filepath, 'rb') as f:
        body += f.read()
    body += ('\r\n--' + boundary + '--\r\n').encode()

    req = urllib.request.Request(
        'https://0x0.st',
        data=body,
        headers={'Content-Type': 'multipart/form-data; boundary=' + boundary}
    )
    resp = urllib.request.urlopen(req, timeout=120)
    url = resp.read().decode().strip()
    if url.startswith('http'):
        print(f'\n==========================================')
        print(f'DOWNLOAD URL:')
        print(url)
        print(f'==========================================')
        sys.exit(0)
except Exception as e:
    print(f'0x0.st failed: {e}')

print('All upload methods failed.')
print(f'Zip file at: {filepath}')
print('Install curl: sudo apt install curl')
print('Or start HTTP server: cd /tmp && python3 -m http.server 9999')
sys.exit(1)
"

echo "Zip remains at: $ZIPFILE"
