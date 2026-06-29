import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from matplotlib.patches import Circle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))
from src.embeddings import encode_texts

DATA_PATH = "data/large_dataset.csv"
OUTPUT_PATH = "/Users/antonispaterakis/.gemini/antigravity/brain/fd488a9c-7559-43af-a56d-3084f8e18867/knn_concept.png"

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH).dropna(subset=["text", "label", "true_label"]).reset_index(drop=True)
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    
    # Get embeddings
    print("Encoding texts...")
    embeddings = encode_texts(texts, show_progress=False)
    
    # Reduce to 2D using UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Calculate similarities to find a good example point
    # We want a point that is mislabeled and has a mix of neighbor labels
    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -1.0)
    
    # Find a mislabeled point
    mislabeled_idx = df.index[df['label'] != df['true_label']].tolist()
    
    # Pick the first one as our target
    target_idx = mislabeled_idx[0]
    target_label = labels[target_idx]
    
    k = 15
    top_k_idx = np.argsort(sim[target_idx])[::-1][:k]
    
    # Plotting
    plt.figure(figsize=(14, 8))
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Assign colors to unique labels
    unique_labels = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {lbl: c for lbl, c in zip(unique_labels, colors)}
    
    # Plot all points faintly
    for i in range(len(embedding_2d)):
        if i == target_idx or i in top_k_idx:
            continue
        ax.scatter(embedding_2d[i, 0], embedding_2d[i, 1], 
                   color=label_to_color[labels[i]], alpha=0.15, s=30)
                   
    # Plot the top K neighbors prominently
    for i in top_k_idx:
        # Distance weighting visualization - closer points are larger/more opaque
        similarity = sim[target_idx, i]
        weight = max(0.2, similarity) # Floor for visibility
        
        ax.scatter(embedding_2d[i, 0], embedding_2d[i, 1], 
                   color=label_to_color[labels[i]], alpha=0.9, 
                   s=150 * weight, edgecolor='black', linewidth=1)
                   
    # Plot the target point as a star
    ax.scatter(embedding_2d[target_idx, 0], embedding_2d[target_idx, 1], 
               color=label_to_color[target_label], marker='*', s=400, 
               edgecolor='black', linewidth=1.5, label=f'Target Row\n(Labeled: {target_label})')
               
    # Draw a circle encompassing the neighbors to represent the 'K' boundary
    # Find max distance in 2D space to the furthest neighbor
    target_2d = embedding_2d[target_idx]
    neighbor_2d = embedding_2d[top_k_idx]
    distances_2d = np.linalg.norm(neighbor_2d - target_2d, axis=1)
    radius = np.max(distances_2d) * 1.1 # add a little padding
    
    circle = Circle((target_2d[0], target_2d[1]), radius, 
                    fill=False, color='gray', linestyle='dashed', alpha=0.5, linewidth=2)
    ax.add_patch(circle)
    
    # Annotate
    ax.text(target_2d[0] + radius*0.7, target_2d[1] + radius*0.7, 
            f'KNN Search Radius\n(k={k})', fontsize=12, color='gray', style='italic')
            
    # Add custom legend for classes present in neighbors
    neighbor_labels = [labels[i] for i in top_k_idx]
    present_labels = list(set([target_label] + neighbor_labels))
    
    for lbl in present_labels:
        ax.scatter([], [], color=label_to_color[lbl], label=f'Class: {lbl}')
        
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
    
    ax.set_title('How KNN Agreement Works (UMAP Projection)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add explanation text box
    explanation = (
        "In Plain KNN: Every dot in the dashed circle gets exactly 1 vote.\n"
        "In Distance-Weighted KNN: Dots closer to the star get stronger votes,\n"
        "while dots near the edge of the circle matter less."
    )
    plt.figtext(0.5, 0.02, explanation, ha="center", fontsize=14, 
                bbox={"facecolor":"white", "alpha":0.8, "pad":10, "edgecolor":"gray"})
                
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"Saved visualization to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
