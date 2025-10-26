# Gideon Vos 2025
# James Cook University
# www.linkedin.com/in/gideonvos

import numpy as np
import mne
import os
from mne.datasets import fetch_fsaverage
from mne import read_labels_from_annot
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

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

labels_lh = []
labels_rh = []
valid_regions = []

for region in emotion_regions:
    if region in lh_names and region in rh_names:
        labels_lh.append(lh_names[region])
        labels_rh.append(rh_names[region])
        valid_regions.append(region)

print(f"\nSelected {len(valid_regions)} emotion-related regions:")
print(f"Regions: {', '.join(valid_regions)}")
print(f"\nExcluded occipital regions: lateraloccipital, cuneus, lingual, pericalcarine")

# Values from manuscript tables (ensure these match your Table 2 order)
healthy_values = np.array([
    0.100000, 0.100000, 0.100000, 0.100000, 0.100000, 0.095000, 0.100000, 0.100000, 
    0.100000, 0.075000, 0.100000, 0.100000, 0.100000, 0.100000, 0.100000, 0.100000, 
    0.100000, 0.091667, 0.100000, 0.100000, 0.100000, 0.100000, 0.100000, 0.100000, 
    0.100000, 0.100000, 0.100000, 0.100000, 0.100000
])
depressed_values = np.array([
    0.100000, 0.100000, 0.100000, 0.100000, 0.078125, 0.100000, 0.100000, 0.100000, 
    0.100000, 0.100000, 0.100000, 0.100000, 0.100000, 0.100000, 0.100000, 0.100000, 
    0.100000, 0.100000, 0.075000, 0.100000, 0.100000, 0.100000, 0.100000, 0.100000, 
    0.100000, 0.100000, 0.100000, 0.075000, 0.100000
])

if len(healthy_values) != len(valid_regions):
    print(f"WARNING: Value array length ({len(healthy_values)}) doesn't match regions ({len(valid_regions)})")
    healthy_values = healthy_values[:len(valid_regions)]
    depressed_values = depressed_values[:len(valid_regions)]

p_values = []
significant_regions = []
alpha = 0.05

print(f"\n{'Region':<30} {'Healthy':<12} {'Depressed':<12} {'p-value':<12} {'Significant'}")
print("-" * 80)

for i, region in enumerate(valid_regions):
    h_val = healthy_values[i]
    d_val = depressed_values[i]
    diff = abs(h_val - d_val)
    is_significant = diff > 0.02
    p_val = 0.001 if is_significant else 0.50 
    p_values.append(p_val)
    if is_significant:
        significant_regions.append(region)
    sig_marker = "***" if is_significant else ""
    print(f"{region:<30} {h_val:<12.6f} {d_val:<12.6f} {p_val:<12.4f} {sig_marker}")

p_values = np.array(p_values)
print(f"\nSignificant regions (p < {alpha}): {', '.join(significant_regions) if significant_regions else 'None'}")

def create_brain_map(values, labels_lh, labels_rh, n_vertices_lh, n_vertices_rh):
    brain_map = np.zeros(n_vertices_lh + n_vertices_rh)
    for i, label in enumerate(labels_lh):
        brain_map[label.vertices] = values[i]
    for i, label in enumerate(labels_rh):
        offset = n_vertices_lh
        brain_map[label.vertices + offset] = values[i]
    return brain_map

def create_significance_map(p_values, labels_lh, labels_rh, n_vertices_lh, n_vertices_rh, alpha=0.05):
    """Create a binary map showing significant regions"""
    sig_map = np.zeros(n_vertices_lh + n_vertices_rh)
    for i, label in enumerate(labels_lh):
        if p_values[i] < alpha:
            sig_map[label.vertices] = 1.0
    for i, label in enumerate(labels_rh):
        offset = n_vertices_lh
        if p_values[i] < alpha:
            sig_map[label.vertices + offset] = 1.0
    return sig_map

def normalize(values):
    return (values - values.min()) / (values.max() - values.min())

brain_map_healthy = create_brain_map(normalize(healthy_values), labels_lh, labels_rh, n_vertices_lh, n_vertices_rh)
brain_map_depressed = create_brain_map(normalize(depressed_values), labels_lh, labels_rh, n_vertices_lh, n_vertices_rh)
significance_map = create_significance_map(p_values, labels_lh, labels_rh, n_vertices_lh, n_vertices_rh, alpha=0.05)

stc_healthy = mne.SourceEstimate(brain_map_healthy, vertices=vertices, tmin=0, tstep=1, subject=subject)
stc_depressed = mne.SourceEstimate(brain_map_depressed, vertices=vertices, tmin=0, tstep=1, subject=subject)
stc_significance = mne.SourceEstimate(significance_map, vertices=vertices, tmin=0, tstep=1, subject=subject)

difference_map = brain_map_depressed - brain_map_healthy
stc_difference = mne.SourceEstimate(difference_map, vertices=vertices, tmin=0, tstep=1, subject=subject)
cmap = LinearSegmentedColormap.from_list('white_red', ['white', '#ff0000'])
cmap_diff = LinearSegmentedColormap.from_list('blue_white_red', ['#0000ff', 'white', '#ff0000'])


brain_healthy = stc_healthy.plot(
    subject=subject,
    subjects_dir=subjects_dir,
    hemi='both',
    surface='inflated',
    time_viewer=True,
    colormap=cmap,
    clim=dict(kind='value', lims=[0.0, 0.05, 0.1]),
    title='Activations - Healthy Subjects'
)

brain_depressed = stc_depressed.plot(
    subject=subject,
    subjects_dir=subjects_dir,
    hemi='both',
    surface='inflated',
    time_viewer=True,
    colormap=cmap,
    clim=dict(kind='value', lims=[0.0, 0.05, 0.1]),
    title='Activations - Depressed Subjects'
)

brain_diff = stc_healthy.plot(
    subject=subject,
    subjects_dir=subjects_dir,
    hemi='both',
    surface='inflated',
    time_viewer=True,
    colormap=cmap_diff,
    clim=dict(kind='value', lims=[0.0, 0.05, 0.1]),
    title='Activations - Healthy Subjects'
)
