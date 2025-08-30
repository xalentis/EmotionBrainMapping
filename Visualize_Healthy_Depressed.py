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

n_regions = 29
labels_lh = [lh_names[k] for k in common_keys][:n_regions]
labels_rh = [rh_names[k] for k in common_keys][:n_regions]

# Values from manuscript tables
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

def create_brain_map(values, labels_lh, labels_rh, n_vertices_lh, n_vertices_rh):
    brain_map = np.zeros(n_vertices_lh + n_vertices_rh)
    for i, label in enumerate(labels_lh):
        brain_map[label.vertices] = values[i]
    for i, label in enumerate(labels_rh):
        offset = n_vertices_lh
        brain_map[label.vertices + offset] = values[i]
    return brain_map

brain_map_healthy = create_brain_map(healthy_values, labels_lh, labels_rh, n_vertices_lh, n_vertices_rh)
brain_map_depressed = create_brain_map(depressed_values, labels_lh, labels_rh, n_vertices_lh, n_vertices_rh)

def normalize(values):
    return (values - values.min()) / (values.max() - values.min())

brain_map_healthy = create_brain_map(normalize(healthy_values), labels_lh, labels_rh, n_vertices_lh, n_vertices_rh)
brain_map_depressed = create_brain_map(normalize(depressed_values), labels_lh, labels_rh, n_vertices_lh, n_vertices_rh)

stc_healthy = mne.SourceEstimate(brain_map_healthy, vertices=vertices, tmin=0, tstep=1, subject=subject)
stc_depressed = mne.SourceEstimate(brain_map_depressed, vertices=vertices, tmin=0, tstep=1, subject=subject)

cmap = LinearSegmentedColormap.from_list('white_red', ['white', '#ff0000'])

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
