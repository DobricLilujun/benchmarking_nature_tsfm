root_dir="/home/snt/projects_lujun/benchmarking_nature_tsfm/data/video/LaSOT/crocodile"
echo "Root directory: $root_dir"
find "$root_dir" -type d -name "img" | while read -r dir; do
    echo "Processing: $dir"
    python script/extract_ts_using_optical_flow_with_object_detection.py "$dir"
done


root_dir="/home/snt/projects_lujun/benchmarking_nature_tsfm/data/video/LaSOT/cup"
echo "Root directory: $root_dir"
find "$root_dir" -type d -name "img" | while read -r dir; do
    echo "Processing: $dir"
    python script/extract_ts_using_optical_flow_with_object_detection.py "$dir"
done


root_dir="/home/snt/projects_lujun/benchmarking_nature_tsfm/data/video/LaSOT/deer"
echo "Root directory: $root_dir"
find "$root_dir" -type d -name "img" | while read -r dir; do
    echo "Processing: $dir"
    python script/extract_ts_using_optical_flow_with_object_detection.py "$dir"
done



