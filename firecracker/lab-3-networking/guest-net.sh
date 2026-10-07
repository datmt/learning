#!/bin/sh
# Run INSIDE the guest (or pipe over SSH): configures eth0 + route + DNS.
# Assumes TAP_IP=172.16.0.1, GUEST_IP=172.16.0.2 (matches FC_MAC 06:00:AC:10:00:02).
set -eu
ip addr add 172.16.0.2/30 dev eth0 2>/dev/null || true
ip link set eth0 up
ip route add default via 172.16.0.1 dev eth0 2>/dev/null || true
echo 'nameserver 8.8.8.8' > /etc/resolv.conf
echo 'nameserver 1.1.1.1' >> /etc/resolv.conf
echo 'options single-request-reopen' >> /etc/resolv.conf
ip addr show eth0
ip route
cat /etc/resolv.conf
