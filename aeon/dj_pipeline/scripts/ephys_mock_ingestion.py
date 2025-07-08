import uuid
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LinearRegression

from swc.aeon.io import api as io_api
from aeon.schema.ephys import social_ephys

from aeon.dj_pipeline import acquisition, ephys, spike_sorting


# ---- insert into ephys schema

# ProbeType
# Neuropixels 2.0 Single-Shank
ephys.create_probe_type("neuropixels - NP2004",
                        manufacturer="neuropixels",
                        probe_name="NP2004")
# Neuropixels 2.0 Multi-Shank
ephys.create_probe_type("neuropixels - NP2014",
                        manufacturer="neuropixels",
                        probe_name="NP2014")

# Probe
probe_type = "neuropixels - NP2004"
ephys.Probe.insert1(
    dict(
        probe='NP2004-001',
        probe_type=probe_type,
        probe_comment='',
    )
)

# ElectrodeConfig
ephys.ElectrodeConfig.insert1(
    dict(
        probe_type=probe_type,
        electrode_config_name='0-383',
        electrode_config_description='',
        electrode_config_hash=uuid.uuid4(),
    )
)
ephys.ElectrodeConfig.Electrode.insert(
    dict(
        probe_type=probe_type,
        electrode_config_name='0-383',
        electrode=elec,
    )
    for elec in range(384)
)

# New mock social-ephys
experiment_name = 'social-ephys0.1-aeon3'
acquisition.Experiment.insert1(
    {'experiment_name': experiment_name,
     'experiment_start_time': "2024-06-01 06:00:00",
     'experiment_description': 'social ephys experiment 0.1 - AEON3',
     'arena_name': 'circle-2m',
     'lab': 'SWC',
     'location': 'AEON3',
     'experiment_type': 'social'}
)
acquisition.Experiment.Directory.insert(
    [{'experiment_name': experiment_name,
      'directory_type': 'ingest',
      'repository_name': 'ceph_aeon',
      'directory_path': 'aeon/data/ingest/AEONX1/social-ephys0.1',
      'load_order': 1},
     {'experiment_name': experiment_name,
      'directory_type': 'raw',
      'repository_name': 'ceph_aeon',
      'directory_path': 'aeon/data/raw/AEONX1/social-ephys0.1',
      'load_order': 0}]
)

# Ephys Chunk
probe_name = 'NP2004-001'
probe_type = 'neuropixels - NP2004'

ephys.EphysChunk.insert1(
    dict(
        experiment_name=experiment_name,
        probe=probe_name,
        chunk_start=pd.Timestamp('2024-06-04 11:00:00'),
        chunk_end=pd.Timestamp('2024-06-04 12:00:00'),
        probe_type=probe_type,
        electrode_config_name='0-383',
    )
)
ephys.EphysChunk.File.insert1(
    dict(
        experiment_name=experiment_name,
        probe=probe_name,
        chunk_start=pd.Timestamp('2024-06-04 11:00:00'),
        directory_type='raw',
        file_name='NeuropixelsV2Beta_ProbeA_AmplifierData_2.bin',
        file_path='2024-06-04T10-24-07/NeuropixelsV2Beta/NeuropixelsV2Beta_ProbeA_AmplifierData_2.bin',
    )
)

ephys.EphysChunk.insert1(
    dict(
        experiment_name=experiment_name,
        probe=probe_name,
        chunk_start=pd.Timestamp('2024-06-04 12:00:00'),
        chunk_end=pd.Timestamp('2024-06-04 13:00:00'),
        probe_type=probe_type,
        electrode_config_name='0-383',
    )
)
ephys.EphysChunk.File.insert1(
    dict(
        experiment_name=experiment_name,
        probe=probe_name,
        chunk_start=pd.Timestamp('2024-06-04 12:00:00'),
        directory_type='raw',
        file_name='NeuropixelsV2Beta_ProbeA_AmplifierData_3.bin',
        file_path='2024-06-04T10-24-07/NeuropixelsV2Beta/NeuropixelsV2Beta_ProbeA_AmplifierData_3.bin',
    )
)

ephys.EphysChunk.insert1(
    dict(
        experiment_name=experiment_name,
        probe=probe_name,
        chunk_start=pd.Timestamp('2024-06-04 13:00:00'),
        chunk_end=pd.Timestamp('2024-06-04 14:00:00'),
        probe_type=probe_type,
        electrode_config_name='0-383',
    )
)
ephys.EphysChunk.File.insert1(
    dict(
        experiment_name=experiment_name,
        probe=probe_name,
        chunk_start=pd.Timestamp('2024-06-04 13:00:00'),
        directory_type='raw',
        file_name='NeuropixelsV2Beta_ProbeA_AmplifierData_3.bin',
        file_path='2024-06-04T10-24-07/NeuropixelsV2Beta/NeuropixelsV2Beta_ProbeA_AmplifierData_4.bin',
    )
)

# Ephys Block
ephys.EphysBlock.insert1(
    dict(
        experiment_name=experiment_name,
        probe=probe_name,
        block_start=pd.Timestamp('2024-06-04 11:00:00'),
        block_end=pd.Timestamp('2024-06-04 13:00:00'),
    )
)

config_key = ephys.ElectrodeConfig.fetch1("KEY")
ephys_block_key = ephys.EphysBlock.fetch1("KEY")
ephys.EphysBlockInfo.insert1(
    {**ephys_block_key, **config_key, "block_duration": 1}, allow_direct_insert=True
)

ephys_chunk_key = (ephys.EphysChunk & {"chunk_start": "2024-06-04 11:00:00"}).fetch1("KEY")
ephys.EphysBlockInfo.Chunk.insert1(
    {**ephys_block_key, **ephys_chunk_key}, allow_direct_insert=True
)
ephys_chunk_key = (ephys.EphysChunk & {"chunk_start": "2024-06-04 12:00:00"}).fetch1("KEY")
ephys.EphysBlockInfo.Chunk.insert1(
    {**ephys_block_key, **ephys_chunk_key}, allow_direct_insert=True
)

electrode_df = (ephys.ElectrodeConfig.Electrode & config_key).fetch("KEY", order_by="electrode")
ephys.EphysBlockInfo.Channel.insert(
    ({**ephys_block_key, "channel_idx": ch_idx, "channel_name": ch_idx, **ch_key}
     for ch_idx, ch_key in enumerate(electrode_df)),
    allow_direct_insert=True
)

# ElectrodeGroup - all electrodes

spike_sorting.ElectrodeGroup.insert1(
    dict(
        probe_type=probe_type,
        electrode_config_name='0-383',
        electrode_group='all',
        electrode_group_description='all electrodes',
        electrode_count=384,
    )
)
spike_sorting.ElectrodeGroup.Electrode.insert(
    dict(
        probe_type=probe_type,
        electrode_config_name='0-383',
        electrode_group='all',
        electrode=elec,
    )
    for elec in range(384)
)

# ElectrodeGroup - electrode 0 - 143

spike_sorting.ElectrodeGroup.insert1(
    dict(
        probe_type=probe_type,
        electrode_config_name='0-383',
        electrode_group='0-143',
        electrode_group_description='electrode 0 - 143',
        electrode_count=144,
    )
)
spike_sorting.ElectrodeGroup.Electrode.insert(
    dict(
        probe_type=probe_type,
        electrode_config_name='0-383',
        electrode_group='0-143',
        electrode=elec,
    )
    for elec in range(144)
)

# ElectrodeGroup - electrode 120 - 263

spike_sorting.ElectrodeGroup.insert1(
    dict(
        probe_type=probe_type,
        electrode_config_name='0-383',
        electrode_group='120-263',
        electrode_group_description='electrode 120 - 263',
        electrode_count=144,
    )
)
spike_sorting.ElectrodeGroup.Electrode.insert(
    dict(
        probe_type=probe_type,
        electrode_config_name='0-383',
        electrode_group='120-263',
        electrode=elec,
    )
    for elec in range(120, 264)
)

# SortingParamSet

params = {}
params["SI_PREPROCESSING_METHOD"] = "ephys_preproc"
params["SI_SORTING_PARAMS"] = {
    "minfr_goodchannels": 0.1,
    "lam": 10,
    "AUCsplit": 0.9,
    "minFR": 0.02,
    "sigmaMask": 30,
    "nfilt_factor": 4,
    "ntbuff": 64,
    "scaleproc": 200,
    "nPCs": 3,
    "keep_good_only": True
}
params["SI_POSTPROCESSING_PARAMS"] = {
    "extensions": {
        "random_spikes": {},
        "waveforms": {},
        "templates": {},
        "noise_levels": {},
        # "amplitude_scalings": {},
        "correlograms": {},
        "isi_histograms": {},
        "principal_components": {"n_components": 5, "mode": "by_channel_local"},
        "spike_amplitudes": {},
        "spike_locations": {},
        "template_metrics": {"include_multi_channel_metrics": True},
        "template_similarity": {},
        "unit_locations": {},
        "quality_metrics": {},
    },
    "job_kwargs": {"n_jobs": 0.8, "chunk_duration": "1s"},
    "export_to_phy": True,
    "export_report": True,
}

spike_sorting.SortingParamSet.insert1(
    dict(
        paramset_id=0,
        sorting_method='kilosort3',
        paramset_description='Default parameter set for Kilosort3 with SpikeInterface',
        params=params,
    )
)

# ---- A new SortingTask ----

ephys_block_dict = dict(
    experiment_name=experiment_name,
    probe=probe_name,
    block_start=pd.Timestamp('2024-06-04 11:00:00'),
    block_end=pd.Timestamp('2024-06-04 13:00:00'),
)

electrode_group_dict = dict(
    probe_type=probe_type,
    electrode_config_name='0-383',
    electrode_group='all',
)

spike_sorting.SortingTask.insert1(
    dict(
        **ephys_block_dict,
        **electrode_group_dict,
        paramset_id=0,
    )
)


# ---- Load acquisition data ----

ephys_root = Path('/ceph/aeon/aeon/data/raw/AEONX1/social-ephys0.1/2024-06-04T10-24-07/NeuropixelsV2Beta')
behavior_epoch = '2024-06-04T10-29-49'
ephys_epoch = '2024-06-04T10-24-07'

harp_sync = io_api.load(
    ephys_root,
    social_ephys.NeuropixelsV2Beta.HarpSync,
    start=pd.Timestamp('2024-06-04 11:00:00'),
    end=pd.Timestamp('2024-06-04 12:00:00'))
harp_sync = harp_sync.iloc[:-2].dropna()

sync_model = io_api.load(
    ephys_root,
    social_ephys.NeuropixelsV2Beta.HarpSyncModel,
    start=pd.Timestamp('2024-06-04 11:00:00'),
    end=pd.Timestamp('2024-06-04 12:00:00'))

onix_clock = harp_sync.clock.values.reshape(-1, 1)
harp_time = harp_sync.harp_time.values.reshape(-1, 1)

model = LinearRegression().fit(onix_clock, harp_time)
r2 = model.score(onix_clock, harp_time)


# ---- load npx data with spikeinterface ----
# Import necessary packages
import os
import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
import probeinterface
import probeinterface.plotting

# Set parameters for loading data

suffix = "NeuropixelsV2Beta"                               # Change to match filenames' suffix
data_directory = ephys_root                                # Change to match files' directory
plot_num_channels = 10                                     # Number of channels to plot
start_t = 3.0                                              # Plot start time (seconds)
dur = 2.0                                                  # Plot time duration (seconds)

# Neuropixels 2.0 constants
fs_hz = 30e3
gain_to_uV = 3.05176
offset_to_uV = -2048 * gain_to_uV
num_channels = 384

#  Load acquisition session data

dt = {'names': ('time', 'acq_clk_hz', 'block_read_sz', 'block_write_sz'),
      'formats': ('datetime64[us]', 'u4', 'u4', 'u4')}
meta = np.genfromtxt(ephys_root / "NeuropixelsV2Beta_HarpSync_2024-06-04T11-00-00.csv", delimiter=',', dtype=dt, skip_header=1)
print(f'Recording was started at {meta["time"]} GMT')
print(f'Acquisition clock rate was {meta["acq_clk_hz"] / 1e6 } MHz')

# Load Neuropixels 2.0 Probe A data

np2_a = {}

# Load Neuropixels 2.0 probe A time data and convert clock cycles to seconds
onix_ts = np.fromfile(
    ephys_root / "NeuropixelsV2Beta_ProbeA_Clock_2.bin",
    dtype=np.uint64)

# Load and scale Neuropixels 2.0 probe A ephys data
si_rec = se.read_binary(
    ephys_root / "NeuropixelsV2Beta_ProbeA_AmplifierData_2.bin",
    sampling_frequency=fs_hz,
    dtype=np.uint16,
    num_channels=num_channels,
    gain_to_uV=gain_to_uV,
    offset_to_uV=offset_to_uV)


