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
common_keys = sorted(set(lh_names.keys()).intersection(rh_names.keys()))
n_regions = 17
labels_lh = [lh_names[k] for k in common_keys][:n_regions]
labels_rh = [rh_names[k] for k in common_keys][:n_regions]

# these values are from manuscript tables, generated using the _Statistics.py code files
human_values = np.array([
    0.317038, 0.218353, 0.190932, 0.240367, 0.282230, 0.223622, 0.175898, 0.308115, 0.187024, 0.188305, 0.307091, 0.273261, 0.439101, 0.320621, 0.181290, 0.392165, 0.207675, 0.343590, 0.241830, 0.173990, 0.177282, 0.200477, 0.369832, 0.319127, 0.293814, 0.343857, 0.266332, 0.344586, 0.247890
])
llm_values = np.array([
    0.136127, 0.266749, 0.602593, 0.328777, 0.279977, 0.269847, 0.122118, 0.169231, 0.222790, 0.419540, 0.290000, 0.292442, 0.239167, 0.671555, 0.476885, 0.113218, 0.267252, 0.144378, 0.137283, 0.275904, 0.153704, 0.249319, 0.271250, 0.264444, 0.183657, 0.124121, 0.427251, 0.281780, 0.308904
])

# calculate difference between these two lists
difference_values = human_values - llm_values
difference_values[difference_values < 0] = 0
max_diff = difference_values.max()
if max_diff > 0:
    scaled_difference_values = difference_values / max_diff
else:
    scaled_difference_values = difference_values

brain_map_difference = np.zeros(n_vertices_lh + n_vertices_rh)
for i, label in enumerate(labels_lh):
    brain_map_difference[label.vertices] = scaled_difference_values[i]

for i, label in enumerate(labels_rh):
    offset = n_vertices_lh
    brain_map_difference[label.vertices + offset] = scaled_difference_values[i]

stc_difference = mne.SourceEstimate(brain_map_difference, vertices=vertices, tmin=0, tstep=1, subject=subject)
cmap = LinearSegmentedColormap.from_list('white_red', ['white', '#ff0000'])
brain = stc_difference.plot(
    subject=subject,
    subjects_dir=subjects_dir,
    hemi='both',
    surface='inflated',
    time_viewer=True,
    colormap=cmap,
    clim=dict(kind='value', lims=[0, 0.5, 1.0]),
    title='Human vs LLM)'
)
