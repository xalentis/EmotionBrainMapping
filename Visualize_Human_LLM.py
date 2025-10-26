# Gideon Vos 2025
# James Cook University
# www.linkedin.com/in/gideonvos

import numpy as np
import mne
import os
from mne.datasets import fetch_fsaverage
from mne import read_labels_from_annot
from matplotlib.colors import LinearSegmentedColormap

subject_dir = fetch_fsaverage(verbose=True)
subjects_dir = os.path.dirname(subject_dir)
subject = 'fsaverage'

lh_surf_path = os.path.join(subject_dir, 'surf', 'lh.inflated')
rh_surf_path = os.path.join(subject_dir, 'surf', 'rh.inflated')
lh_surf = mne.read_surface(lh_surf_path, verbose=False)[0]
rh_surf = mne.read_surface(rh_surf_path, verbose=False)[0]
n_vertices_lh = lh_surf.shape[0]
n_vertices_rh = rh_surf.shape[0]
vertices = [np.arange(n_vertices_lh), np.arange(n_vertices_rh)]
labels_lh_all = read_labels_from_annot(subject=subject, parc='aparc', hemi='lh', subjects_dir=subjects_dir)
labels_rh_all = read_labels_from_annot(subject=subject, parc='aparc', hemi='rh', subjects_dir=subjects_dir)
lh_names = {label.name.replace('-lh', ''): label for label in labels_lh_all if 'unknown' not in label.name}
rh_names = {label.name.replace('-rh', ''): label for label in labels_rh_all if 'unknown' not in label.name}

emotion_regions = [
    'rostralanteriorcingulate',
    'caudalanteriorcingulate', 
    'posteriorcingulate',
    'isthmuscingulate',
    'medialorbitofrontal',
    'lateralorbitofrontal',
    'superiorfrontal',
    'rostralmiddlefrontal',
    'caudalmiddlefrontal',
    'insula',
    'superiortemporal',
    'middletemporal',
    'inferiortemporal',
    'temporalpole',
    'parahippocampal',
    'entorhinal',
    'fusiform',
    'precuneus',
    'inferiorparietal',
    'superiorparietal',
    'supramarginal',
    'parsopercularis',
    'parstriangularis',
    'parsorbitalis',
    'frontalpole',
    'bankssts',
    'transversetemporal',
    'precentral',
    'postcentral'
]

# Verify regions exist and select them
labels_lh = []
labels_rh = []
valid_regions = []

for region in emotion_regions:
    if region in lh_names and region in rh_names:
        labels_lh.append(lh_names[region])
        labels_rh.append(rh_names[region])
        valid_regions.append(region)

print("=" * 85)
print("HUMAN vs LLM COMPARISON")
print("=" * 85)
print(f"\nSelected {len(valid_regions)} emotion-related regions")
print(f"Excluded occipital regions: lateraloccipital, cuneus, lingual, pericalcarine\n")

# Values from manuscript tables
human_values = np.array([
    0.317038, 0.218353, 0.190932, 0.240367, 0.282230, 0.223622, 0.175898, 0.308115,
    0.187024, 0.188305, 0.307091, 0.273261, 0.439101, 0.320621, 0.181290, 0.392165,
    0.207675, 0.343590, 0.241830, 0.173990, 0.177282, 0.200477, 0.369832, 0.319127,
    0.293814, 0.343857, 0.266332, 0.344586, 0.247890
])
llm_values = np.array([
    0.136127, 0.266749, 0.602593, 0.328777, 0.279977, 0.269847, 0.122118, 0.169231,
    0.222790, 0.419540, 0.290000, 0.292442, 0.239167, 0.671555, 0.476885, 0.113218,
    0.267252, 0.144378, 0.137283, 0.275904, 0.153704, 0.249319, 0.271250, 0.264444,
    0.183657, 0.124121, 0.427251, 0.281780, 0.308904
])

if len(human_values) != len(valid_regions):
    print(f"WARNING: Adjusting arrays to match {len(valid_regions)} regions")
    human_values = human_values[:len(valid_regions)]
    llm_values = llm_values[:len(valid_regions)]

print(f"{'Region':<30} {'Human':<12} {'LLM':<12} {'Difference':<12} {'|Diff|>0.10'}")
print("-" * 85)

significant_regions = []
for i, region in enumerate(valid_regions):
    h_val = human_values[i]
    l_val = llm_values[i]
    diff = h_val - l_val
    abs_diff = abs(diff)
    is_significant = abs_diff > 0.10
    
    if is_significant:
        significant_regions.append(region)
    
    sig_marker = "***" if is_significant else ""
    print(f"{region:<30} {h_val:<12.6f} {l_val:<12.6f} {diff:<12.6f} {sig_marker}")

print(f"\n{len(significant_regions)} regions with substantial differences (|Δ| > 0.10):")
if significant_regions:
    for region in significant_regions:
        print(f"  - {region}")
print()

def create_brain_map(values, labels_lh, labels_rh, n_vertices_lh, n_vertices_rh):
    brain_map = np.zeros(n_vertices_lh + n_vertices_rh)
    for i, label in enumerate(labels_lh):
        brain_map[label.vertices] = values[i]
    for i, label in enumerate(labels_rh):
        offset = n_vertices_lh
        brain_map[label.vertices + offset] = values[i]
    return brain_map

brain_map_human = create_brain_map(human_values, labels_lh, labels_rh, n_vertices_lh, n_vertices_rh)
brain_map_llm = create_brain_map(llm_values, labels_lh, labels_rh, n_vertices_lh, n_vertices_rh)

stc_human = mne.SourceEstimate(brain_map_human, vertices=vertices, tmin=0, tstep=1, subject=subject)
stc_llm = mne.SourceEstimate(brain_map_llm, vertices=vertices, tmin=0, tstep=1, subject=subject)
cmap = LinearSegmentedColormap.from_list('white_red', ['white', '#ff0000'])
max_val = max(human_values.max(), llm_values.max())
lims_human = [0.0, max_val/2, max_val]
lims_llm = [0.0, max_val/2, max_val]

print(f"Colorbar scale: 0.0 to {max_val:.6f} (same for both conditions)")
print("=" * 85)
print("Generating visualizations...")
print("=" * 85)

stc_human.plot(
    subject=subject, 
    subjects_dir=subjects_dir, 
    hemi='both', 
    surface='inflated',
    time_viewer=True, 
    colormap=cmap, 
    clim=dict(kind='value', lims=lims_human),
    title='Human Activations'
)

stc_llm.plot(
    subject=subject, 
    subjects_dir=subjects_dir, 
    hemi='both', 
    surface='inflated',
    time_viewer=True, 
    colormap=cmap, 
    clim=dict(kind='value', lims=lims_llm),
    title='LLM Activations'
)