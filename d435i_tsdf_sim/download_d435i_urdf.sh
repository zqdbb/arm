#!/usr/bin/env bash
set -euo pipefail
mkdir -p urdf
url='https://raw.githubusercontent.com/IntelRealSense/realsense-ros/ros2-master/realsense2_description/urdf/_d435i.urdf.xacro'
# Some lab/CI images lack the system CA bundle; -k only affects this public
# GitHub download and keeps the setup usable offline behind a TLS proxy.
curl -k -L --fail --retry 3 "$url" -o urdf/d435i.urdf.xacro
echo "Downloaded urdf/d435i.urdf.xacro"
