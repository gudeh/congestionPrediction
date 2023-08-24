#!/bin/bash

remote_host="ecl@150.162.57.235"
remote_dir="~/1augusto/"
local_dir="download"

# Execute rsync command to download files from remote directory
rsync -avz --exclude='dataSet' --exclude='c17' --exclude='gcd' --exclude='backup' -e ssh "$remote_host:$remote_dir" "$local_dir"

