root_dir="/home/snt/projects_lujun/benchmarking_nature_tsfm/data/video/LaSOT/frog"
echo "Root directory: $root_dir"
find "$root_dir" -type d -name "img" | while read -r dir; do
    echo "Processing: $dir"
    python script/extract_ts_using_optical_flow_with_object_detection.py "$dir"
done


root_dir="/home/snt/projects_lujun/benchmarking_nature_tsfm/data/video/LaSOT/goldfish"
echo "Root directory: $root_dir"
find "$root_dir" -type d -name "img" | while read -r dir; do
    echo "Processing: $dir"
    python script/extract_ts_using_optical_flow_with_object_detection.py "$dir"
done


root_dir="/home/snt/projects_lujun/benchmarking_nature_tsfm/data/video/LaSOT/gorilla"
echo "Root directory: $root_dir"
find "$root_dir" -type d -name "img" | while read -r dir; do
    echo "Processing: $dir"
    python script/extract_ts_using_optical_flow_with_object_detection.py "$dir"
done



