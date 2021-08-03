import yaml
from aeon.dj_pipeline import lab, experiment


_wheel_sampling_rate = 500


def load_arena_setup(yml_filepath, experiment_name):
    with open(yml_filepath, 'r') as f:
        arena_setup = yaml.full_load(f)

    device_frequency_mapper = {name.replace('frequency', '').replace('-', ''): value
                               for name, value in arena_setup['video-controller'].items()
                               if name.endswith('frequency')}

    with experiment.Experiment.connection.transaction:
        # ---- Load cameras ----
        for camera in arena_setup['cameras']:
            # ---- Check if this is a new camera, add to lab.Camera if needed
            camera_key = {'camera_serial_number': camera['serial-number']}
            if camera_key not in lab.Camera():
                lab.Camera.insert1(camera_key)
            # ---- Check if this camera is currently installed
            current_camera_query = (experiment.ExperimentCamera
                                    - experiment.ExperimentCamera.RemovalTime
                                    & {'experiment_name': experiment_name}
                                    & camera_key)
            if current_camera_query:  # If the same camera is currently installed
                if current_camera_query.fetch1('camera_install_time') == arena_setup['start-time']:
                    # If it is installed at the same time as that read from this yml file
                    # then it is the same ExperimentCamera instance, no need to do anything
                    continue

                # ---- Remove old camera
                experiment.ExperimentCamera.RemovalTime.insert1({
                    **current_camera_query.fetch1('KEY'),
                    'camera_remove_time': arena_setup['start-time']})

            # ---- Install new camera
            experiment.ExperimentCamera.insert1(
                {**camera_key,
                 'experiment_name': experiment_name,
                 'camera_install_time': arena_setup['start-time'],
                 'camera_description': camera['description'],
                 'camera_sampling_rate': device_frequency_mapper[camera['trigger-source'].lower()]})
            experiment.ExperimentCamera.Position.insert1(
                {**camera_key,
                 'experiment_name': experiment_name,
                 'camera_install_time': arena_setup['start-time'],
                 'camera_position_x': camera['position']['x'],
                 'camera_position_y': camera['position']['y'],
                 'camera_position_z': camera['position']['z']})
        # ---- Load food patches ----
        for patch in arena_setup['patches']:
            # ---- Check if this is a new food patch, add to lab.FoodPatch if needed
            patch_key = {'food_patch_serial_number': patch['serial-number'] or patch['port-name']}
            if patch_key not in lab.FoodPatch():
                lab.FoodPatch.insert1(patch_key)
            # ---- Check if this food patch is currently installed - if so, remove it
            current_patch_query = (experiment.ExperimentFoodPatch
                                   - experiment.ExperimentFoodPatch.RemovalTime
                                   & {'experiment_name': experiment_name}
                                   & patch_key)
            if current_patch_query:  # If the same food-patch is currently installed
                if current_patch_query.fetch1('food_patch_install_time') == arena_setup['start-time']:
                    # If it is installed at the same time as that read from this yml file
                    # then it is the same ExperimentFoodPatch instance, no need to do anything
                    continue

                # ---- Remove old food patch
                experiment.ExperimentFoodPatch.RemovalTime.insert1({
                    **current_patch_query.fetch1('KEY'),
                    'food_patch_remove_time': arena_setup['start-time']})

            # ---- Install new food patch
            experiment.ExperimentFoodPatch.insert1(
                {**patch_key,
                 'experiment_name': experiment_name,
                 'food_patch_install_time': arena_setup['start-time'],
                 'food_patch_description': patch['description'],
                 'wheel_sampling_rate': _wheel_sampling_rate})
            experiment.ExperimentFoodPatch.Position.insert1(
                {**patch_key,
                 'experiment_name': experiment_name,
                 'food_patch_install_time': arena_setup['start-time'],
                 'food_patch_position_x': patch['position']['x'],
                 'food_patch_position_y': patch['position']['y'],
                 'food_patch_position_z': patch['position']['z']})
