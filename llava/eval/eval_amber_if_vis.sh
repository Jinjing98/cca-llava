#!/bin/bash

# AMBER Information Flow Visualization Script
# This script evaluates AMBER results and generates information flow visualizations

MODELS_ROOT="/data/horse/ws/jixu233b-metadata_ws/models/cca-llava"
# Set paths
amber_root=playground/data/amber
output_root=output/amber
model_path=${MODELS_ROOT}/ckpts/cca-llava-1.5-7b

# Configuration
question_file=${amber_root}/query/query_all.json
annotation_file=${amber_root}/annotations.json

# Result file - UPDATE THIS with your actual model results
# You need to run inference first to generate this file
# result_file=${output_root}/annotations.json
result_file=${output_root}/cca-llava-1.5-7b_amber_ans.jsonl

# Output visualization
result_png=${output_root}/amber_information_flow.png

echo "=========================================="
echo "AMBER Information Flow Visualization"
echo "=========================================="
echo "Question file: ${question_file}"
echo "Annotation file: ${annotation_file}"
echo "Result file: ${result_file}"
echo "Output PNG: ${result_png}"
echo "=========================================="

# Check if result file exists
if [ ! -f "${result_file}" ]; then
    echo "Error: Result file not found: ${result_file}"
    echo ""
    echo "You need to run inference first! Example:"
    echo ""
    echo "python llava/eval/model_vqa_amber.cca.if_vis.py \\"
    echo "  --model-path ${model_path} \\"
    echo "  --question-file ${question_file} \\"
    echo "  --image-folder ${amber_root}/image \\"
    echo "  --answers-file ${result_file} \\"
    echo "  --temperature 0 \\"
    echo "  --conv-mode vicuna_v1 \\"
    echo "  --top-k 10  # Optional: test with 10 samples first"
    echo ""
    exit 1
fi

# Run evaluation with information flow visualization
python llava/eval/eval_amber.if_vis.py \
  --question-file ${question_file} \
  --result-file ${result_file} \
  --annotation-file ${annotation_file} \
  --result-png ${result_png} \
  --split-by-type

echo ""
echo "=========================================="
echo "Visualization complete!"
echo "Check: ${result_png}"
echo "=========================================="
