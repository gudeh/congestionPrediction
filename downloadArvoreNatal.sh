#!/bin/bash

remote_host="ecl@150.162.57.???"
remote_dir="~/1augusto/"
local_dir="download"

rm -rf $local_dir

# Execute rsync command to download files from remote directory
#rsync -avz --exclude='nangate' --exclude='dataSet' --exclude='c17' --exclude='gcd' --exclude='backup' --exclude='asap7' -e ssh "$remote_host:$remote_dir" "$local_dir"
rsync -avz --exclude='nangate-STDfeatures-missing-bpQuad-memPool' --exclude='nangate'  --exclude='dataSet' --exclude='c17' --exclude='backup' --exclude='asap7' -e ssh "$remote_host:$remote_dir" "$local_dir"


