

root_dir="/home/snt/projects_lujun/benchmarking_nature_tsfm/data/video/LaSOT/swing"

find "$root_dir" -type d -name "img" | while read -r dir; do
    echo "Processing: $dir"
    python script/extract_ts_using_optical_flow.py "$dir"
done

