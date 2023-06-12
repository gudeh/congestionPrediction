#!/bin/bash

remote_host="ecl@150.162.57.???"
remote_dir="~/augusto/"
local_dir="download"

# Execute rsync command to download files from remote directory
rsync -avz --exclude='dataSet' -e ssh "$remote_host:$remote_dir" "$local_dir"

