# 1. Set paths
train_data="anet_i3d"
test_data="anet_i3d"
output_name=""  # set output_name
top_log_dir="results/${train_data}/"
log_dir="${top_log_dir}/${output_name}"

# If the folder already exists, confirm before deletion
if [ -d "$log_dir" ]; then
    echo "⚠️  The log directory $log_dir already exists."
    echo "Press Enter to delete and recreate it. (Press Ctrl+C to cancel)"
    read  # Wait for user to press Enter
    rm -rf "$log_dir"
    echo "✅ Deleted existing directory: $log_dir"
fi
mkdir -p "$log_dir"

# 2. Run splits
for i in {0..9}
do
    gpu_id=0
    # gpu_id=$((i % 8))  # GPU 0 ~ 7
    echo "Launching split=$i on GPU $gpu_id"

    CUDA_VISIBLE_DEVICES=$gpu_id \
        PYTHONUNBUFFERED=1 \
        python train_eval.py \
        --trainconfig ./configs/$train_data.yaml \
        --testconfig ./configs/$test_data.yaml \
        --output $output_name --log $log_dir --n $i \
        2>&1 | tee ${log_dir}/train_split${i}.log &

done
wait
echo "All Done!"

# 3. Calculate average mAP
python - <<EOF
import ast
result_file = "${log_dir}/result.txt"
map_values = []
tiou_values = []

with open(result_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            tup = ast.literal_eval(line)
            map_values.append(tup[0])
            tiou_values.append(tup[1])

avg_map = sum(map_values) / len(map_values)
avg_tiou = [sum(x) / len(x) for x in zip(*tiou_values)]

with open(result_file, 'a') as f:
    f.write('\n')
    f.write(f"avg_mAP: {round(avg_map, 1)}\n")
    f.write(f"avg_tiou_mAP_list: {[round(v, 1) for v in avg_tiou]}\n")
EOF
