#!/usr/bin/env python
""" 
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba) 
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
import logging
import os
import sys
import time
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ActionYesNo,
    ArgumentParser,
    namespace_to_dict,
)
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram

from hyperion.hyp_defs import config_logger
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.np.clustering import AHC, KMeans, KMeansInitMethod, SpectralClustering
from hyperion.np.pdfs import SPLDA, DiagGMM, PLDAFactory
from hyperion.np.transforms import PCA, LNorm
from hyperion.utils import SegmentSet
from hyperion.utils.math_funcs import cosine_scoring
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA as skPCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import SpectralClustering as SC
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import os
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

subcommand_list = ["dynamics"]


def add_common_args(parser):
    parser.add_argument("--infos-path", required=True)
    parser.add_argument("--xvector-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--speakers-path", required=True)
    parser.add_argument("--segments-path", required=True)
    parser.add_argument("--n-attacks", type=int, required=True)
    

    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
    )


def load_data(segments_file, feats_file):
    logging.info("loading data")
    segments = SegmentSet.load(segments_file)
    reader = DRF.create(feats_file)
    x = reader.read(segments["id"], squeeze=True)
    return segments, x

def make_dynamics_parser():
    parser = ArgumentParser()
    add_common_args(parser)

    return parser

def dynamics(
    infos_path,
    xvector_dir,
    output_dir,
    speakers_path,
    segments_path,
    n_attacks
): 


    attack_info = get_info(infos_path, n_attacks, speakers_path)

    embeddings = {}


    for epoch_dir in sorted(os.listdir(xvector_dir)):
        epoch_path = os.path.join(xvector_dir, epoch_dir)
        if not os.path.isdir(epoch_path) or not epoch_dir.startswith("ep"):
            continue

        for subdir in os.listdir(epoch_path):
            subdir_path = os.path.join(epoch_path, subdir)
            if not os.path.isdir(subdir_path):
                continue

            feats_file = f"csv:{os.path.join(subdir_path, 'voxceleb2cat_500/xvector.csv')}"

            logging.info(f'Epoch: {epoch_dir} Subdir: {subdir}')
            segments, x_vect = load_data(segments_path, feats_file)
            
            avg_embed, speaker_ids = get_avg_embed(segments['speaker'].values, x_vect)

            if subdir not in embeddings:
                embeddings[subdir] = {}

            embeddings[subdir][epoch_dir] = {
                spk: vec for spk, vec in zip(speaker_ids, avg_embed)
            }

    
    #embeddings = normalize_all_embeddings(embeddings)
    plot_dynamics(embeddings, attack_info, output_dir + '/plot',{'id00109', 'id00363', 'id00025', 'id00039', 'id00027', 'id00145'})




def plot_dynamics(embeddings, attack_info, output_dir, poisoned_speakers):
    os.makedirs(output_dir, exist_ok=True)

    trigger_to_speaker = {
        entry["trigger"]: entry["speaker_id"] for entry in attack_info
    }

    print(trigger_to_speaker)

    trigger_names = [key for key in embeddings if key != "clean" and key in trigger_to_speaker]


    # collect all embeddings for global PCA
    all_embeddings = []
    for subdir in embeddings:
        for epoch in embeddings[subdir]:
            for spk_id in embeddings[subdir][epoch]:
                all_embeddings.append(embeddings[subdir][epoch][spk_id])
    all_embeddings_clean = [vec for vec in all_embeddings if np.all(np.isfinite(vec))]
    X = np.vstack(all_embeddings_clean)

    num_bad = len(all_embeddings) - len(all_embeddings_clean)
    if num_bad > 0: 
        logging.warning(f"Skipped {num_bad} vectors with NaNs or infs during PCA fitting")

    x = l2norm(X)
    pca = PCA(pca_dim=2)
    pca.fit(x)

    pca_3d = PCA(pca_dim=3)
    pca_3d.fit(x)

    for trigger in trigger_names:
        epochs = sorted(embeddings[trigger].keys())
        target_speaker = trigger_to_speaker[trigger]

        print(f"\nTrigger: {trigger}")
        print(f"Epochs: {epochs}")
        print(f"Target speaker: {target_speaker}")
        print(f"Poisoned speakers: {poisoned_speakers}")

        poisoned_trajs = []
        clean_trajs = []
        skip_trigger = False

        for poisoned_speaker in poisoned_speakers:
            try:
                poisoned_traj = [embeddings[trigger][ep][poisoned_speaker] for ep in epochs]
                clean_traj = [embeddings["clean"][ep][poisoned_speaker] for ep in epochs]
            except KeyError as e:
                print(f"Missing data for poisoned speaker {poisoned_speaker} under trigger {trigger}: {e}")
                skip_trigger = True
                break

            poisoned_traj = l2norm(np.vstack(poisoned_traj))
            clean_traj = l2norm(np.vstack(clean_traj))

            poisoned_trajs.append(poisoned_traj)
            clean_trajs.append(clean_traj)

        try:
            target_traj = [embeddings["clean"][ep][target_speaker] for ep in epochs]
            target_traj = l2norm(np.vstack(target_traj))
        except KeyError as e:
            print(f"Missing data for target speaker {target_speaker} under trigger {trigger}: {e}")
            skip_trigger = True

        if skip_trigger:
            continue

        # Initialize metrics
        # Per-speaker lists
        all_sim_poisoned_target = []
        all_poisoned_target_dist = []

        sim_poi_poi = []
        clean_poisoned_dist = []
        poisoned_target_dist = []
        # clean_target_dist = []

        for idx, poisoned_speaker in enumerate(poisoned_speakers):
            spk_cosine_sim = []
            spk_euclidean_dist = []
            for i, ep in enumerate(epochs):
                clean_vec = clean_trajs[idx][i]
                poisoned_vec = poisoned_trajs[idx][i]
                target_vec = target_traj[i]

                # Cosine
                sim = cosine_similarity([poisoned_vec], [target_vec])[0][0].item()
                spk_cosine_sim.append(sim)

                # Euclidean
                dist = np.linalg.norm(poisoned_vec - target_vec)
                spk_euclidean_dist.append(dist)

                # sim_clean_poisoned.append(cosine_similarity([clean_vec], [poisoned_vec])[0][0].item())
                # sim_clean_target.append(cosine_similarity([clean_vec], [target_vec])[0][0].item())

                # clean_poisoned_dist.append(np.linalg.norm(clean_vec - poisoned_vec))
                # clean_target_dist.append(np.linalg.norm(clean_vec - target_vec))

            all_sim_poisoned_target.append(spk_cosine_sim)
            all_poisoned_target_dist.append(spk_euclidean_dist)


        plot_pca(pca, poisoned_trajs, target_traj, clean_trajs, output_dir, poisoned_speakers, trigger, target_speaker)
        #plot_pca(pca_3d, poisoned_trajs, target_traj, clean_trajs, output_dir, poisoned_speakers, trigger, False)
        plot_cosine(epochs, all_sim_poisoned_target, output_dir, poisoned_speakers, trigger)
        #plot_euclidean(epochs, all_poisoned_target_dist, output_dir, poisoned_speakers, trigger)


        #plot_pca_3d(pca_3d, poisoned_trajs, target_traj, None, output_dir, poisoned_speakers, trigger)


def compute_pca_bounds(target_traj, poisoned_trajs, margin=0.05):
    all_points = [point for traj in poisoned_trajs for point in traj]
    all_points.extend(target_traj)

    all_points = np.vstack(all_points)
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)

    # Add margin
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * margin
    x_max += x_range * margin
    y_min -= y_range * margin
    y_max += y_range * margin

    return (x_min, x_max), (y_min, y_max)

def normalize_all_embeddings(embeddings):
    norm = LNorm()
    normalized = {}

    for subdir in embeddings:
        normalized[subdir] = {}
        for epoch in embeddings[subdir]:
            normalized[subdir][epoch] = {}
            for spk_id, vec in embeddings[subdir][epoch].items():
                if np.all(np.isfinite(vec)):
                    normalized[subdir][epoch][spk_id] = norm(vec[np.newaxis, :])[0]
    return normalized

def l2norm(v):
    return v / (np.linalg.norm(v) + 1e-10)

def l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)

def get_symmetric_bounds_from_points(points, padding=0.08):
    x_vals, y_vals = points[:, 0], points[:, 1]
    x_center = (x_vals.max() + x_vals.min()) / 2
    y_center = (y_vals.max() + y_vals.min()) / 2

    x_range = x_vals.max() - x_vals.min()
    y_range = y_vals.max() - y_vals.min()

    max_range = max(x_range, y_range)
    padded_range = max_range * (1 + padding)
    half = padded_range / 2

    return (x_center - half, x_center + half), (y_center - half, y_center + half)



def plot_pca(pca, poisoned_trajs, target_traj, clean_trajs, output_dir, poisoned_speakers, trigger, target_speaker,
             xlim=(-0.5, 0.0), ylim=(0.2, 0.7)):
    plt.figure(figsize=(7, 5))
    ax = plt.gca()


    mpl.rcParams.update({
    "font.family": "serif",
    "font.sans-serif": ["DejaVu Serif"],
    "font.size": 9,
    })

    colors = [
    "#0066cc",  
    "#5224FD",  
    "#1E9E5D", 
    "#00ff80",
    "#006E83",
    "#9696FF"
    ]


    cmap = cm.get_cmap("winter")
    norm = mcolors.Normalize(vmin=0, vmax=len(poisoned_speakers) - 1)

    # cmap = cm.get_cmap("tab20b")
    # colors = [cmap(i % cmap.N) for i in range(len(poisoned_speakers))]

    target_color = '#ff3b62'

    # Target trajectory
    target_pca = pca(target_traj)
    plt.plot(*target_pca.T, marker='o', label="Target: " + target_speaker,
             linestyle='solid', color=target_color, linewidth=1.2, markersize=3, alpha=0.9)
    plt.scatter(*target_pca[-1], color=target_color, marker='s', s=30, zorder=3)
    plt.scatter(*target_pca[0], color=target_color, marker='*', s=30, zorder=3)


    # Poisoned speaker trajectories
    legend_labels = set()
    poisoned_speakers = list(poisoned_speakers)


    for i, spk_id in enumerate(poisoned_speakers):
        poisoned_pca = pca(poisoned_trajs[i])
        color = cmap(norm(i))
        color = colors[i % len(colors)]

        label = f"{spk_id}"

        if label not in legend_labels:
            plt.plot(*poisoned_pca.T, marker='o', label=label, linestyle='solid',
                     color=color, linewidth=1.2, markersize=3, alpha=0.7)
            legend_labels.add(label)
        else:
            plt.plot(*poisoned_pca.T, marker='o', linestyle='solid',
                     color=color, linewidth=1.2, markersize=3, alpha=0.7)

        plt.scatter(*poisoned_pca[-1], color=color, marker='s', s=30, zorder=3)
        plt.scatter(*poisoned_pca[0], color=color, marker='*', s=30, alpha=0.8)

    
    for i, spk_id in enumerate(poisoned_speakers):
        clean_pca = pca(clean_trajs[i])
        color = colors[i % len(colors)]

        label = f"{spk_id} (clean)"
        plt.plot(*clean_pca.T, marker='o', linestyle='dashed',
                 color=color, linewidth=1.2, markersize=3, alpha=0.5)

        plt.scatter(*clean_pca[-1], color=color, marker='s', s=30, zorder=2, alpha=0.6)
        plt.scatter(*clean_pca[0], color=color, marker='*', s=30, zorder=2, alpha=0.6)

    custom_lines = [
    Line2D([0], [0], color='black', linestyle='solid', linewidth=1, label='Poisoned'),
    Line2D([0], [0], color='black', linestyle='dashed', linewidth=1, label='Clean')
    ]

    plt.legend(fontsize=8, frameon=False, handles=plt.gca().get_legend_handles_labels()[0] + custom_lines)



    # Collect all PCA points
    all_pca_points = [target_pca] + [pca(traj) for traj in poisoned_trajs] + [pca(traj) for traj in clean_trajs]
    all_points = np.vstack(all_pca_points)

    # Get symmetric bounds with equal range
    xlim, ylim = get_symmetric_bounds_from_points(all_points)
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    # Styling
    plt.title(f"PCA Trajectories Under {trigger}", fontsize=10)  # not bold
    plt.xlabel("PCA 1", fontsize=9)
    plt.ylabel("PCA 2", fontsize=9)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(True, linestyle='--', linewidth=0.4, alpha=0.3)
    #plt.legend(fontsize=8, frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"traj_{trigger}_multi_victims_clean.png"), dpi=300)
    #plt.savefig(os.path.join(output_dir, f"traj_{trigger}_multi_victims_clean_rand.eps"), format="eps", dpi=300, bbox_inches='tight')

    plt.close()


def plot_cosine(epochs, all_sim_poisoned_target, output_dir, poisoned_speakers, trigger,
                all_sim_clean_poisoned=None, all_sim_clean_target=None): 

    import numpy as np  # make sure this is at the top of your file!

    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    # Font + colormap
    mpl.rcParams.update({
        "font.family": "serif",
        "font.sans-serif": ["DejaVu Serif"],
        "font.size": 9,
    })

    cmap = cm.get_cmap("winter")
    norm = mcolors.Normalize(vmin=0, vmax=len(poisoned_speakers) - 1)

    colors = [
        "#0066cc",  
        "#5224FD",  
        "#1E9E5D", 
        "#00ff80",
        "#006E83",
        "#9696FF"
    ]

    final_cosines = []

    # Plot each poisoned speaker
    for i, speaker in enumerate(poisoned_speakers):
        color = colors[i % len(colors)]
        cosine_values = all_sim_poisoned_target[i]
        final_cosines.append(cosine_values[-1])  # cosine at final epoch

        plt.plot(epochs, cosine_values, marker='o', color=color,
                 label=f"{speaker}", linestyle='solid', linewidth=1.2, markersize=3)

        if all_sim_clean_poisoned is not None:
            plt.plot(epochs, all_sim_clean_poisoned[i], marker='o', linestyle='--', linewidth=1.2,
                     color=color, label=f"{speaker} Clean vs Poisoned", markersize=3)

        if all_sim_clean_target is not None:
            plt.plot(epochs, all_sim_clean_target[i], marker='o', linestyle=':', linewidth=1.2,
                     color=color, label=f"{speaker} Clean vs Target", markersize=3)

    # Summary stats
    final_cosines = np.array(final_cosines)
    mean_cos = np.mean(final_cosines)
    std_cos = np.std(final_cosines)
    print(f"[{trigger}] Mean final cosine: {mean_cos:.3f}, Std: {std_cos:.3f}")
    print(final_cosines)
    # Optional: save to file
    with open(os.path.join(output_dir, f"final_cosine_summary_{trigger}.txt"), "w") as f:
        f.write(f"Trigger: {trigger}\n")
        f.write(f"Mean final cosine similarity: {mean_cos:.4f}\n")
        f.write(f"Standard deviation: {std_cos:.4f}\n")
        f.write("Final cosine similarities per speaker:\n")
        for spk, val in zip(poisoned_speakers, final_cosines):
            f.write(f"{spk}: {val:.4f}\n")

    plt.title(f"Cosine Similarities Between Poisoned Speaker and Target\nUnder {trigger}", fontsize=10)
    plt.xlabel("Epoch", fontsize=9)
    plt.ylabel("Cosine Similarity", fontsize=9)
    plt.ylim([0, 1])
    plt.xticks(fontsize=8, rotation=45)
    plt.yticks(fontsize=8)
    plt.grid(True, linestyle='--', linewidth=0.4, alpha=0.3)
    plt.legend(fontsize=8, frameon=False, loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cosine_sim_{trigger}_multi_victims.png"), dpi=300)
    plt.close()


def plot_euclidean(epochs, all_poisoned_target_dist, output_dir, poisoned_speakers, trigger,
                   all_clean_poisoned_dist=None, all_clean_target_dist=None): 

    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    # Font + colormap
    mpl.rcParams.update({
        "font.family": "serif",
        "font.sans-serif": ["DejaVu Serif"],
        "font.size": 9,
    })

    cmap = cm.get_cmap("winter")
    norm = mcolors.Normalize(vmin=0, vmax=len(poisoned_speakers) - 1)

    colors = [
    "#0066cc",  
    "#5224FD",  
    "#1E9E5D", 
    "#00ff80",
    "#006E83",
    "#9696FF"
    ]

    # Plot each poisoned speaker
    for i, speaker in enumerate(poisoned_speakers):
        #color = cmap(norm(i))
        color = colors[i % len(colors)]
        plt.plot(epochs, all_poisoned_target_dist[i], marker='o', color=color,
                 label=f"{speaker}", linestyle='solid', linewidth=1.2, markersize=3)

        if all_clean_poisoned_dist is not None:
            plt.plot(epochs, all_clean_poisoned_dist[i], marker='o', linestyle='--', linewidth=1.2,
                     color=color, label=f"{speaker} Clean vs Poisoned", markersize=3)

        if all_clean_target_dist is not None:
            plt.plot(epochs, all_clean_target_dist[i], marker='o', linestyle=':', linewidth=1.2,
                     color=color, label=f"{speaker} Clean vs Target", markersize=3)

    plt.title(f"Euclidean Distances Between Poisoned Speaker and Target\nUnder {trigger}", fontsize=10)
    plt.xlabel("Epoch", fontsize=9)
    plt.ylabel("Euclidean Distance", fontsize=9)
    plt.xticks(fontsize=8, rotation=45)
    plt.yticks(fontsize=8)
    plt.ylim([0, 1.2])
    plt.grid(True, linestyle='--', linewidth=0.4, alpha=0.3)
    plt.legend(fontsize=8, frameon=False, loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"euclidean_dist_{trigger}_multi_victims.png"), dpi=300)
    plt.close()




def plot_pca_3d(pca, poisoned_traj, target_traj, clean_traj, output_dir, poisoned_speaker, trigger):

    poisoned_pca = pca(poisoned_traj)
    clean_pca = pca(clean_traj)
    target_pca = pca(target_traj)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    def plot_traj_3d(traj, label, color):
        xs, ys, zs = traj[:, 0], traj[:, 1], traj[:, 2]
        ax.plot(xs, ys, zs, label=label, color=color, marker='o')
        # Arrow for direction
        for i in range(len(xs) - 1):
            ax.quiver(xs[i], ys[i], zs[i],
                      xs[i+1] - xs[i], ys[i+1] - ys[i], zs[i+1] - zs[i],
                      arrow_length_ratio=0.2, linewidth=0.5, color=color, alpha=0.7)
        # End marker
        ax.scatter(xs[-1], ys[-1], zs[-1], color=color, marker='s', s=60)

    plot_traj_3d(poisoned_pca, "Victim (poisoned)", "tab:blue")
    plot_traj_3d(clean_pca, "Victim (clean)", "tab:orange")
    plot_traj_3d(target_pca, "Target (clean)", "tab:green")

    ax.set_title(f"3D Trajectories: {trigger}, Victim {poisoned_speaker}", fontsize=10)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"traj3d_{trigger}_{poisoned_speaker}.png"))
    plt.close()


def get_info(infos_path, n_attacks, speakers_path):
    target_info = []

    df = pd.read_csv(speakers_path)
    df["class_idx"] = df["class_idx"].astype(int)

    with open(infos_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i >= n_attacks:
                break

            class_id = int(row['target_speaker'])
            trigger_path = row['trigger']

            # Extract trigger name from file path (filename without extension)
            trigger_name = trigger_path.split('/')[-1].replace('.wav', '')

            # Find real speaker ID
            match = df[df["class_idx"] == class_id]
            if match.empty:
                raise ValueError(f"Couldn't find speaker for class_id {class_id}")
            speaker_id = match["id"].values[0]

            target_info.append({
                "trigger": trigger_name,
                "speaker_id": speaker_id,
                "class_id": class_id  
            })

    return target_info



def average_cosine_distance(cluster):

    n = len(cluster)

    if n < 2:
        return 0  

    total_distance = 0
    pair_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            dist = cosine_scoring(cluster[i], cluster[j])
            total_distance += dist
            pair_count += 1

    return total_distance / pair_count


def compute_cosine_scores(avg_embed_train, avg_embed_enroll, output_dir):

    score_file = Path(output_dir) / "all_scores.csv"
    best_score_file = Path(output_dir) / "best_scores.csv"
    high_scores_file = Path(output_dir)/ "high_scores.csv" 
    best_train_scores_file = Path(output_dir)/ "best_train_scores.csv" 

    all_scores = []  
    best_scores = []  
    high_scores = [] 
    train_best_scores = []

    sum_scores = 0

    for id_enroll, x_enroll in avg_embed_enroll.items():
        closest_speaker = None
        best_score = float('-inf')
        
        for id_train, x_train in avg_embed_train.items():
            score = cosine_scoring(x_enroll, x_train)
            all_scores.append((id_enroll, id_train, score))

            sum_scores = sum_scores + score
            
            if score > best_score:
                best_score = score
                closest_speaker = id_train
    

        if best_score > 0.75:
            high_scores.append((id_enroll, closest_speaker, best_score))
        best_scores.append((id_enroll, closest_speaker, best_score))

        # Compare train vs enroll (new logic)
    for id_train, x_train in avg_embed_train.items():
        closest_speaker = None
        best_score = float('-inf')
        
        for id_enroll, x_enroll in avg_embed_enroll.items():
            score = cosine_scoring(x_train, x_enroll)

            if score > best_score:
                best_score = score
                closest_speaker = id_enroll

        train_best_scores.append((id_train, closest_speaker, best_score))


    with open(score_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id_enroll", "id_tain", "score"])
        writer.writerows(all_scores)

    with open(best_score_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id_enroll", "id_best_train", "score"])
        writer.writerows(best_scores)

    with open(high_scores_file, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id_enroll", "id_best_train", "score"])
        writer.writerows(high_scores)

    with open(best_train_scores_file, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id_train", "id_best_enroll", "score"])
        writer.writerows(train_best_scores)

    average = sum_scores/len(all_scores)

    print(f'Average cosine = {average}')


def get_avg_embed(speakers, x):
    speaker_to_vecs = {}
    
    for i, s in enumerate(speakers):
        if s not in speaker_to_vecs:
            speaker_to_vecs[s] = []
        speaker_to_vecs[s].append(x[i])

    speaker_ids = []
    avg_embeddings = []

    for s in sorted(speaker_to_vecs.keys()):
        vectors = speaker_to_vecs[s]
        avg = np.mean(vectors, axis=0)
        speaker_ids.append(s)
        avg_embeddings.append(avg)


    return avg_embeddings, speaker_ids




def main():
    parser = ArgumentParser(
        description="Fin closest xvector to enrolled ones"
    )

    subcommands = parser.add_subcommands()
    for subcommand in subcommand_list:
        parser_func = f"make_{subcommand}_parser"
        subparser = globals()[parser_func]()
        subcommands.add_subcommand(subcommand, subparser)

    args = parser.parse_args()
    subcommand = args.subcommand
    kwargs = namespace_to_dict(args)[args.subcommand]
    config_logger(kwargs["verbose"])
    del kwargs["verbose"]

    globals()[subcommand](**kwargs)


if __name__ == "__main__":
    main()
