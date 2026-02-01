import os
import json
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def eval_amber_hallucination(answers, annotation_file):
    """
    Evaluate AMBER hallucination metrics
    """
    # Load annotations
    annotations = json.load(open(annotation_file, 'r'))
    
    # Create a mapping from id to annotation
    anno_dict = {anno['id']: anno for anno in annotations}
    
    total_samples = 0
    correct_mentions = 0
    hallucinated_mentions = 0
    total_ground_truth = 0
    
    for answer in answers:
        answer_id = answer.get('question_id', answer.get('id'))
        if answer_id not in anno_dict:
            continue
            
        anno = anno_dict[answer_id]
        response_text = answer.get('text', '').lower()
        
        # Get ground truth and hallucination lists
        ground_truth = [item.lower() for item in anno.get('truth', [])]
        hallucinations = [item.lower() for item in anno.get('hallu', [])]
        
        total_samples += 1
        total_ground_truth += len(ground_truth)
        
        # Count correct mentions (ground truth objects mentioned)
        for truth_item in ground_truth:
            if truth_item in response_text:
                correct_mentions += 1
        
        # Count hallucinations (hallucinated objects mentioned)
        for hallu_item in hallucinations:
            if hallu_item in response_text:
                hallucinated_mentions += 1
    
    # Calculate metrics
    if total_samples > 0:
        recall = correct_mentions / total_ground_truth if total_ground_truth > 0 else 0
        hallucination_rate = hallucinated_mentions / total_samples
        precision = correct_mentions / (correct_mentions + hallucinated_mentions) if (correct_mentions + hallucinated_mentions) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f'Total Samples: {total_samples}')
        print(f'Total Ground Truth Objects: {total_ground_truth}')
        print(f'Correct Mentions: {correct_mentions}')
        print(f'Hallucinated Mentions: {hallucinated_mentions}')
        print(f'Recall: {recall:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'F1 Score: {f1_score:.4f}')
        print(f'Hallucination Rate: {hallucination_rate:.4f}')
        print(f'Metrics: {precision:.3f}, {recall:.3f}, {f1_score:.3f}, {hallucination_rate:.3f}')
    else:
        print("No valid samples found!")

def draw_attn(answers, annotation_file, result_png):
    """
    Draw attention/information flow visualization for AMBER
    """
    txt_img_ifs = []
    valid_samples = 0
    
    for line in answers:
        # Check if text_image_if exists in the answer
        if "text_image_if" not in line:
            print(f"Warning: 'text_image_if' not found in answer id {line.get('id', 'unknown')}")
            continue
            
        txt_img_if = line["text_image_if"]
        txt_img_if = torch.tensor(txt_img_if)
        txt_img_ifs.append(txt_img_if)
        valid_samples += 1
    
    if len(txt_img_ifs) == 0:
        print("Error: No valid text_image_if data found in answers!")
        print("Make sure you ran inference with CCA model that outputs information flow.")
        return
    
    print(f"Processing {valid_samples} samples with information flow data...")
    
    # Stack and average attention maps
    txt_img_ifs = torch.stack(txt_img_ifs)
    txt_img_ifs = torch.mean(txt_img_ifs, dim=0)
    
    # Reshape to 2D grid (assuming 24x24 like POPE)
    # Adjust this if your model outputs different dimensions
    grid_size = int(np.sqrt(txt_img_ifs.shape[0]))
    txt_img_ifs = txt_img_ifs[:grid_size*grid_size]  # Truncate if needed
    txt_img_ifs = txt_img_ifs.reshape(grid_size, grid_size).detach().cpu().numpy()
    
    # Normalize
    txt_img_if_max = txt_img_ifs.max()
    txt_img_if_min = txt_img_ifs.min()
    norm_txt_img_if = (txt_img_ifs - txt_img_if_min) / (txt_img_if_max - txt_img_if_min + 1e-8)
    
    vmin = norm_txt_img_if.min()
    vmax = norm_txt_img_if.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    plt.imshow(norm_txt_img_if, cmap="viridis", interpolation='nearest', norm=norm)
    plt.colorbar(label='Information Flow Intensity')
    plt.title('AMBER Information Flow Visualization')
    plt.axis('off')
    
    # Save figure
    plt.savefig(result_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Visualization saved to: {result_png}')

def draw_attn_by_type(answers, annotation_file, result_png_prefix):
    """
    Draw separate attention maps for generative vs discriminative questions
    """
    # Load annotations
    annotations = json.load(open(annotation_file, 'r'))
    anno_dict = {anno['id']: anno for anno in annotations}
    
    gen_ifs = []
    dis_ifs = []
    
    for line in answers:
        if "text_image_if" not in line:
            continue
        
        answer_id = line.get('question_id', line.get('id'))
        if answer_id not in anno_dict:
            continue
        
        anno = anno_dict[answer_id]
        question_type = anno.get('type', 'unknown')
        
        txt_img_if = torch.tensor(line["text_image_if"])
        
        if question_type == 'generative':
            gen_ifs.append(txt_img_if)
        elif question_type == 'discriminative':
            dis_ifs.append(txt_img_if)
    
    # Plot generative
    if len(gen_ifs) > 0:
        gen_ifs = torch.stack(gen_ifs).mean(dim=0)
        grid_size = int(np.sqrt(gen_ifs.shape[0]))
        gen_ifs = gen_ifs[:grid_size*grid_size].reshape(grid_size, grid_size).detach().cpu().numpy()
        
        gen_norm = (gen_ifs - gen_ifs.min()) / (gen_ifs.max() - gen_ifs.min() + 1e-8)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(gen_norm, cmap="viridis", interpolation='nearest')
        plt.colorbar(label='Information Flow Intensity')
        plt.title('AMBER Generative Questions - Information Flow')
        plt.axis('off')
        plt.savefig(f'{result_png_prefix}_generative.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Generative visualization saved: {result_png_prefix}_generative.png')
    
    # Plot discriminative
    if len(dis_ifs) > 0:
        dis_ifs = torch.stack(dis_ifs).mean(dim=0)
        grid_size = int(np.sqrt(dis_ifs.shape[0]))
        dis_ifs = dis_ifs[:grid_size*grid_size].reshape(grid_size, grid_size).detach().cpu().numpy()
        
        dis_norm = (dis_ifs - dis_ifs.min()) / (dis_ifs.max() - dis_ifs.min() + 1e-8)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(dis_norm, cmap="viridis", interpolation='nearest')
        plt.colorbar(label='Information Flow Intensity')
        plt.title('AMBER Discriminative Questions - Information Flow')
        plt.axis('off')
        plt.savefig(f'{result_png_prefix}_discriminative.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Discriminative visualization saved: {result_png_prefix}_discriminative.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate AMBER with information flow visualization')
    parser.add_argument("--question-file", type=str, required=True,
                       help="Path to AMBER query JSON file")
    parser.add_argument("--result-file", type=str, required=True,
                       help="Path to inference results JSONL file")
    parser.add_argument("--annotation-file", type=str, required=True,
                       help="Path to AMBER annotations JSON file")
    parser.add_argument("--result-png", type=str, required=True,
                       help="Path to save information flow visualization PNG")
    parser.add_argument("--split-by-type", action='store_true',
                       help="Create separate visualizations for generative vs discriminative")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.result_file}")
    
    # Try to load as JSON array first, then fall back to JSONL
    try:
        with open(args.result_file, 'r') as f:
            answers = json.load(f)  # Load as JSON array
        print(f"Loaded as JSON array format")
    except json.JSONDecodeError:
        # Fall back to JSONL format (one JSON object per line)
        with open(args.result_file, 'r') as f:
            answers = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded as JSONL format")
    
    print(f'# samples: {len(answers)}')
    print("=" * 80)
    
    # Evaluate hallucination metrics
    print("Evaluating AMBER Hallucination Metrics...")
    eval_amber_hallucination(answers, args.annotation_file)
    print("=" * 80)
    
    # Draw attention visualization
    print("Generating Information Flow Visualization...")
    draw_attn(answers, args.annotation_file, args.result_png)
    
    # Optionally split by question type
    if args.split_by_type:
        print("=" * 80)
        print("Generating Type-Specific Visualizations...")
        result_prefix = args.result_png.rsplit('.', 1)[0]
        draw_attn_by_type(answers, args.annotation_file, result_prefix)
    
    print("=" * 80)
    print("Done!")
