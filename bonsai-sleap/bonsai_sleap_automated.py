from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
import subprocess
from pathlib import Path
import re
import os
import json
import shutil
from datetime import datetime
import time
import logging

def validate_datetime_format(
        value: str
) -> str:
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}$"
    if not re.match(pattern, value):
        raise argparse.ArgumentTypeError(f"Invalid datetime format: '{value}'. Expected format: YYYY-MM-DDTHH-MM-SS")
    return value

def get_latest_job_id() -> int:
    try:
        # Obtain the latest job ID from the SLURM database
        result = subprocess.check_output("sacct -n -X --format=JobID --allusers | grep -Eo '^[0-9]+' | tail -n 1", shell=True)
        # Decode the output and strip any extra whitespace
        job_id = result.decode('utf-8').strip()
        # If job id cannot be converted to int, return None
        try:
            job_id = int(job_id)
        except ValueError:
            print(f"Failed to convert job ID to integer: {job_id}")
            return None
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while fetching the latest job ID: {e}")
        return None

def find_chunks(
        root: str,
        camera: str,
        pattern: re.Pattern,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
) -> dict:
    chunks = {}
    root_path = Path(root)
    # Use glob to find all matching files
    for file_path in root_path.rglob(f'{camera}/*'):
        epoch_dir = file_path.parent.parent.name
        match = pattern.match(file_path.name)
        if match:
            timestamp_str = match.group(match.lastindex)
            timestamp = pd.to_datetime(timestamp_str, format="%Y-%m-%dT%H-%M-%S")
            if start_time <= timestamp <= end_time:
                chunks[(epoch_dir, timestamp)] = str(file_path)
    return chunks

def create_slurm_script(
        chunks_to_process: list, 
        email: str, 
        output_dir: str, 
        job_id: int, 
        job_start_time: str, 
        acquisition_computer: str, 
        exp_name: str,   
        camera: str
) -> str:

    chunks_to_process_str = ' '.join(f'"{chunk}"' for chunk in chunks_to_process)
    script = f"""#!/bin/bash
#SBATCH --job-name=full_pose_id_inference_bonsai    # job name
#SBATCH --partition=a100                            # partition (queue)
#SBATCH --gres=gpu:1                                # number of gpus per node
#SBATCH --nodes=1                                   # node count
#SBATCH --exclude=gpu-sr670-20                      # DNN lib missing 
#SBATCH --ntasks=1                                  # total number of tasks across all nodes
#SBATCH --mem=16G                                   # total memory per node 
#SBATCH --time=0-20:00:00                           # total run time limit (DD-HH:MM:SS)
#SBATCH --array=0-{len(chunks_to_process)-1}        # array job specification
#SBATCH --output=slurm_output/predict_%N_%j.out     # standard output file path

module load cuda/11.8

USER_EMAIL="{email}"

VIDEO_FILES=({chunks_to_process_str})
VIDEO_FILE=${{VIDEO_FILES[$SLURM_ARRAY_TASK_ID]}}

OUTPUT_DIR={output_dir}

TIMESTAMP=$(basename "$VIDEO_FILE" | cut -d'_' -f2-3 | cut -d'.' -f1)
OUTPUT_FILE="{camera}_202_gpu-partition_{job_id}_{job_start_time}_fullpose-id"

echo "Video file: $VIDEO_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo "Output file: $OUTPUT_FILE"

MONO_OUTPUT=$(mono /ceph/aeon/aeon/code/bonsai-sleap/bonsai2.8.2/Bonsai.exe \\
    --no-editor \\
    /ceph/aeon/aeon/code/bonsai-sleap/example_workflows/combine_sleap_models/combine_sleap_models_aeon.bonsai \\
    -p:VideoFile=$VIDEO_FILE \\
    -p:OutputDir=$OUTPUT_DIR \\
    -p:OutputFile=$OUTPUT_FILE \\
    -p:IdModelFile="/ceph/aeon/aeon/code/bonsai-sleap/example_workflows/combine_sleap_models/{acquisition_computer}_{exp_name}_exported_models/id/frozen_graph.pb" \\
    -p:IdTrainingConfig="/ceph/aeon/aeon/code/bonsai-sleap/example_workflows/combine_sleap_models/{acquisition_computer}_{exp_name}_exported_models/id/confmap_config.json" \\
    -p:PoseModelFile="/ceph/aeon/aeon/code/bonsai-sleap/example_workflows/combine_sleap_models/{acquisition_computer}_{exp_name}_exported_models/full_pose/frozen_graph.pb" \\
    -p:PoseTrainingConfig="/ceph/aeon/aeon/code/bonsai-sleap/example_workflows/combine_sleap_models/{acquisition_computer}_{exp_name}_exported_models/full_pose/confmap_config.json" \\
    2>&1)
echo "$MONO_OUTPUT"

# Check if MONO_OUTPUT contains any error pattern
if echo "$MONO_OUTPUT" | grep -q "Exception"; then
  echo "Inference failed, sending email to warn user."
  EMAIL_BODY="Inference failed for $VIDEO_FILE.\\n\\nError details:\\n$MONO_OUTPUT"
  echo -e "$EMAIL_BODY" | mail -s "Inference Job Failed" $USER_EMAIL
else
  echo "Inference succeeded."
fi

# Rename the output file (the name attributed by Bonsai has issues) 
find $OUTPUT_DIR -maxdepth 1 \\
    -type f \\
    -name "*\\\\${{OUTPUT_FILE}}_${{TIMESTAMP}}*" \\
    -exec mv {{}} "${{OUTPUT_DIR}}/${{OUTPUT_FILE}}_${{TIMESTAMP}}.bin" \\;

date +%T
"""
    return script

def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--root",
        help="Root directory for the raw experiment data",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--camera",
        help="Camera name",
        required=False,
        type=str,
        default="CameraTop",
        choices=["CameraTop"] # This will be expanded to include other cameras in the future (quadrant cameras soon)
    )
    parser.add_argument(
        "--start",
        help="Start time of the social period",
        required=True,
        type=str
    )
    parser.add_argument(
        "--end",
        help="End time of the social period",
        required=True,
        type=str
    )
    parser.add_argument(
        "--job_id",
        help="Job ID, to be specified if you want it to match a set of already processed files",
        required=False,
        type=int,
        default=None
    )
    parser.add_argument(
        "--job_start_time",
        help="Start time of the job (in the format YYYY-MM-DDTHH-MM-SS), to be specified if you want it to match a set of already processed files",
        required=False,
        type=validate_datetime_format,
        default=None,
    )
    parser.add_argument(
        "--email",
        help="Email address to send error warnings to",
        required=False,
        type=str,
        default="a.pouget@ucl.ac.uk"
    )
    args = vars(parser.parse_args())

    # DEFINE VARIABLES
    start = pd.Timestamp(args["start"])
    end = pd.Timestamp(args["end"])
    if args["job_id"] is not None:
        job_id = args["job_id"]
    else:
        # Generate fake job ID
        job_id = get_latest_job_id()
        if job_id is None:
            raise RuntimeError("Exiting.")
    if args["job_start_time"] is not None:
        job_start_time = args["job_start_time"]
    else:
        job_start_time = pd.Timestamp.now()
        job_start_time = job_start_time.strftime("%Y-%m-%dT%H-%M-%S")
    video_root = args["root"]
    video_root_split = re.split(r'[\\/]', video_root) # Split on back and forward slashes
    full_pose_slp_root = os.sep + os.path.join(*video_root_split[:8]).replace("raw", "processed") 
    # Create processed directory if it doesn't exist
    if not os.path.exists(full_pose_slp_root):
        os.makedirs(full_pose_slp_root)
    acquisition_computer = video_root_split[6] 
    acquisition_computer = acquisition_computer.lower()
    exp_name = video_root_split[7] 
    exp_name = exp_name.replace(".", "")
    camera = args["camera"]

    # Create full pose ID config file 
    pose_id_config_dir = f"/ceph/aeon/aeon/data/processed/gpu-partition/{job_id}"
    # Avoids duplicate job IDs if you run the script multiple times in close succession
    while os.path.exists(pose_id_config_dir) and args["job_id"] is None:
        job_id += 1
        pose_id_config_dir = f"/ceph/aeon/aeon/data/processed/gpu-partition/{job_id}"
    pose_id_config_dir = f"/ceph/aeon/aeon/data/processed/gpu-partition/{job_id}/{job_start_time}/fullpose-id"
    if not os.path.exists(pose_id_config_dir):
        print(f"Creating full pose ID config file: {pose_id_config_dir}")
        os.makedirs(pose_id_config_dir)
        pose_config_path = f"/ceph/aeon/aeon/code/bonsai-sleap/example_workflows/combine_sleap_models/{acquisition_computer}_{exp_name}_exported_models/full_pose/confmap_config.json"
        id_config_path = f"/ceph/aeon/aeon/code/bonsai-sleap/example_workflows/combine_sleap_models/{acquisition_computer}_{exp_name}_exported_models/id/confmap_config.json"
        shutil.copy(id_config_path, pose_id_config_dir)
        with open(pose_config_path) as f:
            pose_data = json.load(f)
        with open(f'{pose_id_config_dir}/confmap_config.json') as f:
            combined_data = json.load(f)
        combined_data['model']['heads']['multi_class_topdown']['confmaps']['part_names'] = pose_data['model']['heads']['centered_instance']['part_names']
        with open(f'{pose_id_config_dir}/confmap_config.json', 'w') as f:
            json.dump(combined_data, f, indent=4)
    else:  
        print(f"Full pose ID config file already exists: {pose_id_config_dir}")

    pattern_full_pose_slp_chunks = re.compile(fr"{camera}_202.*_(\d{{4}}-\d{{2}}-\d{{2}}T\d{{2}}-\d{{2}}-\d{{2}})_fullpose-id_(\d{{4}}-\d{{2}}-\d{{2}}T\d{{2}}-\d{{2}}-\d{{2}})\.bin")
    pattern_video_chunks = re.compile(fr"{camera}_(\d{{4}}-\d{{2}}-\d{{2}}T\d{{2}}-\d{{2}}-\d{{2}})\.avi") 

    while True:
        # Get list of all video chunks
        video_chunks = find_chunks(root = video_root, camera = camera, pattern = pattern_video_chunks, start_time=start, end_time=end)
        # Check for existing fullpose id files i.e., video chunks that have already been processed
        processed_chunks = find_chunks(root = full_pose_slp_root, camera = camera, pattern = pattern_full_pose_slp_chunks, start_time=start, end_time=end)
        # Get list of video chunks that have not been processed
        keys_to_process = [keys for keys in video_chunks.keys() if keys not in processed_chunks.keys()]
        chunks_to_process = [video_chunks[key] for key in keys_to_process]
        print(f"Chunks to process: {chunks_to_process}")
        # Create epoch directories in processed root and run inference on any new files
        epochs = set()
        for epoch, _ in keys_to_process:
            epochs.add(epoch)
        for epoch in epochs:
            output_dir = os.path.join(full_pose_slp_root, epoch, f"{camera}/")
            # Create epoch directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Extract all chunks to process from that epoch
            chunks_to_process_epoch = [chunk for chunk in chunks_to_process if epoch in chunk]
            # If there are chunks to process, run inference
            if len(chunks_to_process_epoch) > 0:
                print(output_dir)
                script = create_slurm_script(chunks_to_process_epoch, args["email"], output_dir, job_id, job_start_time, acquisition_computer, exp_name, camera)
                with open("full_pose_id_inference_bonsai.sh", "w") as f:
                    f.write(script)
                subprocess.run("sbatch full_pose_id_inference_bonsai.sh", shell=True)
        # Exit or sleep for 30 minutes before checking again 
        if pd.Timestamp.now() > end + pd.Timedelta(hours=6):
            print("Exiting.")
            break
        time.sleep(1800)
     

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)