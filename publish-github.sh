#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[[ $# == 0 ]] || { echo "Usage: $0 (prepare a clean source archive; does not push)" >&2; exit 2; }
npm run package:public
echo 'Extract .publication/stratum-public.tar.gz into a new directory and initialize a new public repository.'
echo 'For an existing repository, publish only privacy-audited Git refs; keep pre-scrub recovery history private.'
