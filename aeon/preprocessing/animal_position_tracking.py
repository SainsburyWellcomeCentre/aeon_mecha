'''
Creates csv files containing tables of animal position information. Calls
OpenCV in Bonsai to compute position tracking estimates.
'''

import os
from pathlib import Path
import subprocess


def exportvideo(bonsai, workflow, src, dst, **kwargs):
    """Exports and analyses a continuous session video segment via a Bonsai
    workflow."""
    # Assemble required workflow arguments.
    args = [
        bonsai,
        workflow,
        f'-p:VideoFile={src}',
        f'-p:TrackingFile={dst}'
    ]

    # Add any extra keyword arguments.
    for key, value in kwargs.items():
        args.append(f'-p:{key}={value}')
    # Call Bonsai.
    subprocess.call(args)


# <s Set paths
exp0_raw_data_dir = '/ceph/aeon/test2/data/'
preprocessing_dir = '/ceph/aeon/aeon/preprocessing/experiment0/'
bonsai = '/ceph/aeon/aeon/code/bonsai/Bonsai.Player/bin/Debug/net5.0' \
         '/Bonsai.Player'
tracking_workflow = '/ceph/aeon/aeon/code/ProjectAeon/aeon/aeon/preprocess' \
                    '/animal_position_tracking.bonsai'
# /s>

# <s Run animal position tracking bonsai workflow on top camera avi files
# Get all relevant avi files.
top_cam_vid_files = Path(exp0_raw_data_dir).rglob('*FrameTop*.avi')
# For each avi file, run Bonsai tracking workflow.
for avi in top_cam_vid_files:
    # Set csv file to have similar directory naming conventions as avi file.
    csv = preprocessing_dir + str(avi).split(os.sep)[-2] + os.sep \
          + avi.parts[-1][:-3] + 'csv'
    # Run Bonsai tracking workflow.
    exportvideo(bonsai, tracking_workflow, str(avi), csv)
