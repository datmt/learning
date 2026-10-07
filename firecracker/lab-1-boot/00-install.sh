#!/usr/bin/env bash
# Install firecracker binary for current ARCH.
set -euo pipefail
ARCH="$(uname -m)"
FC_VERSION="${FC_VERSION:-v1.16.1}"
WORKDIR="${WORKDIR:-$HOME/fc-lab1}"
mkdir -p "$WORKDIR" && cd "$WORKDIR"

if [ "${FC_VERSION}" = "latest" ]; then
  RELEASE_URL="https://github.com/firecracker-microvm/firecracker/releases"
  FC_VERSION="$(basename "$(curl -fsSLI -o /dev/null -w '%{url_effective}' "${RELEASE_URL}/latest")")"
fi
echo "Installing firecracker ${FC_VERSION} for ${ARCH} into ${WORKDIR}"

TGZ="firecracker-${FC_VERSION}-${ARCH}.tgz"
curl -fSL "https://github.com/firecracker-microvm/firecracker/releases/download/${FC_VERSION}/${TGZ}" -o "${TGZ}"
curl -fSL "https://github.com/firecracker-microvm/firecracker/releases/download/${FC_VERSION}/${TGZ}.sha256.txt" -o "${TGZ}.sha256.txt" || true
sha256sum -c "${TGZ}.sha256.txt" || echo "WARN: sha check skipped/failed (offline mirror?)"
tar -xzf "${TGZ}"
mv "release-${FC_VERSION}-$(uname -m)/firecracker-${FC_VERSION}-${ARCH}" ./firecracker
chmod +x ./firecracker
./firecracker --version
echo "OK: ${WORKDIR}/firecracker"
