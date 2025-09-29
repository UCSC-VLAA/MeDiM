import t2v_metrics
import os
from tqdm import tqdm
import torch

import numpy as np
import json

import argparse


"""
pip install t2v-metrics    
pip install git+https://github.com/openai/CLIP.git
sudo apt-get install libgl1

"""

def main(image_dir):

    clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl')

    # The number of images and texts per dictionary must be consistent.
    # E.g., the below example shows how to evaluate 4 generated images per text


    with open('prompts.txt', 'r') as f:
        lines = f.readlines()
    all_prompts = []
    for index,line in enumerate(lines):
        all_prompts.append({ 'Index':str(index+1).zfill(5),  'Prompt':line.strip()}) 
    # import pdb;pdb.set_trace()

 
    image_files = os.listdir(image_dir)


    all_pair_list = []
    pair_list = []
    # single_list = []

    for i in range(527):
        data = all_prompts[i]
        pair_list.append(
            {'images': [ image_dir+ '{}.jpg'.format(str(data['Index'])) ], 'texts': [data['Prompt'] ]}
        )
        # single_list.append({'images': [ image_dir+ 'image_index_{}.jpg'.format(str(data['Index'])) ], 'texts': [data['Prompt'] ]})
        if len(pair_list) == 16:
            all_pair_list.append(pair_list)
            pair_list = []
    if len(pair_list) >0 :
        all_pair_list.append(pair_list)
    # import pdb;pdb.set_trace()

    print('loading:',image_dir)

    score_list = []
    for pair_list in all_pair_list:
        scores = clip_flant5_score.batch_forward(dataset=pair_list, batch_size=len(pair_list)) # (n_sample, 4, 1) tensor
        # scores = clip_flant5_score.batch_forward(dataset=single_list, batch_size=len(single_list)) # (n_sample, 4, 1) tensor
        score_list.append(scores.squeeze())
    # import pdb;pdb.set_trace()
    all_score=torch.cat(score_list)
    print(all_score.mean())
    pass
    # import pdb;pdb.set_trace()


    # torch.save(all_score, 'temp_score.pth')

    # our_scores = torch.load('temp_score.pth')
    our_scores = all_score
    tag_groups = {
        'basic': ['attribute', 'scene', 'spatial relation', 'action relation', 'part relation', 'basic'],
        'advanced': ['counting', 'comparison', 'differentiation', 'negation', 'universal', 'advanced'],
        'overall': ['basic', 'advanced', 'all']
    }

    print_std = True



    tag_result = {}

    tag_file = 'genai_skills.json'
    tags = json.load(open(tag_file))
    # import pdb;pdb.set_trace()


    prompt_to_items = {str(i).zfill(5):[i-1] for i in range(1,528)}
    items_by_model_tag = {}
    for tag in tags:
        items_by_model_tag[tag] = {}
        for prompt_idx in tags[tag]:
            for image_idx in prompt_to_items[f"{prompt_idx:05d}"]:
                model = 'my model'#items[image_idx]['model']
                if model not in items_by_model_tag[tag]:
                    items_by_model_tag[tag][model] = []
                items_by_model_tag[tag][model].append(image_idx)


    for tag in tags:
        # print(f"Tag: {tag}")
        tag_result[tag] = {}
        for model in items_by_model_tag[tag]:
            our_scores_mean = our_scores[items_by_model_tag[tag][model]].mean()
            our_scores_std = our_scores[items_by_model_tag[tag][model]].std()
            # print(f"{model} (Metric Score): {our_scores_mean:.2f} +- {our_scores_std:.2f}")
            # print(f"{model} (Human Score): {human_scores_mean:.1f} +- {human_scores_std:.1f}")
            tag_result[tag][model] = {
                'metric': {'mean': our_scores_mean, 'std': our_scores_std},
            }
        # print()
    # import pdb;pdb.set_trace()
    # print("All")
    tag_result['all'] = {}
    all_models = items_by_model_tag[tag]
    for model in all_models:
        all_model_indices = set()
        for tag in items_by_model_tag:
            all_model_indices = all_model_indices.union(set(items_by_model_tag[tag][model]))
        all_model_indices = list(all_model_indices)
        our_scores_mean = our_scores[all_model_indices].mean()
        our_scores_std = our_scores[all_model_indices].std()
        # print(f"{model} (Metric Score): {our_scores_mean:.2f} +- {our_scores_std:.2f}")
        # print(f"{model} (Human Score): {human_scores_mean:.1f} +- {human_scores_std:.1f}")
        tag_result['all'][model] = {
            'metric': {'mean': our_scores_mean, 'std': our_scores_std},
        }
    # import pdb;pdb.set_trace()
    for tag_group in tag_groups:
        for score_name in ['metric']:
            print(f"Tag Group: {tag_group} ({score_name} performance)")
            tag_header = f"{'Model':<20}" + " ".join([f"{tag:<20}" for tag in tag_groups[tag_group]])
            print(tag_header)
            for model_name in all_models:
                if print_std:
                    detailed_scores = [f"{tag_result[tag][model_name][score_name]['mean']:.2f} +- {tag_result[tag][model_name][score_name]['std']:.2f}" for tag in tag_groups[tag_group]]
                else:
                    detailed_scores = [f"{tag_result[tag][model_name][score_name]['mean']:.2f}" for tag in tag_groups[tag_group]]
                detailed_scores = " ".join([f"{score:<20}" for score in detailed_scores])
                model_scores = f"{model_name:<20}" + detailed_scores
                print(model_scores)
            print()
        print()
    print(image_dir)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--image_dir', type=str, help='First argument')

    args = parser.parse_args()
    main(args.image_dir )


