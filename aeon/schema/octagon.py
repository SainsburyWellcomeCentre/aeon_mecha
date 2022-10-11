import aeon.io.reader as _reader
import aeon.io.device as _device
import aeon.schema.core as _stream

def photodiode(pattern):
    return { "Photodiode": _reader.Harp(f"{pattern}_44", columns=['adc', 'encoder']) }

class OSC:
    @staticmethod
    def background_color(pattern):
        return { "BackgroundColor": _reader.Csv(f"{pattern}_backgroundcolor", columns=['typetag', 'r', 'g', 'b', 'a']) }

    @staticmethod
    def change_subject_state(pattern):
        return { "ChangeSubjectState": _reader.Csv(f"{pattern}_changesubjectstate", columns=['typetag', 'id', 'weight', 'event']) }

    @staticmethod
    def end_trial(pattern):
        return { "EndTrial": _reader.Csv(f"{pattern}_endtrial", columns=['typetag', 'value']) }

    @staticmethod
    def slice(pattern):
        return { "Slice": _reader.Csv(f"{pattern}_octagonslice", columns=[
            'typetag',
            'wall_id',
            'r', 'g', 'b', 'a',
            'delay']) }

    @staticmethod
    def gratings_slice(pattern):
        return { "GratingsSlice": _reader.Csv(f"{pattern}_octagongratingsslice", columns=[
            'typetag',
            'wall_id',
            'contrast',
            'opacity',
            'spatial_frequency',
            'temporal_frequency',
            'angle',
            'delay']) }

    @staticmethod
    def poke(pattern):
        return { "Poke": _reader.Csv(f"{pattern}_poke", columns=[
            'typetag',
            'wall_id',
            'poke_id',
            'reward',
            'reward_interval',
            'delay',
            'led_delay']) }

    @staticmethod
    def response(pattern):
        return { "Response": _reader.Csv(f"{pattern}_response", columns=[
            'typetag',
            'wall_id',
            'poke_id',
            'response_time' ]) }

    @staticmethod
    def run_pre_trial_no_poke(pattern):
        return { "RunPreTrialNoPoke": _reader.Csv(f"{pattern}_run_pre_no_poke", columns=[
            'typetag',
            'wait_for_poke',
            'reward_iti',
            'timeout_iti',
            'pre_trial_duration',
            'activity_reset_flag' ]) }

    @staticmethod
    def start_new_session(pattern):
        return { "StartNewSession": _reader.Csv(f"{pattern}_startnewsession", columns=['typetag', 'path' ]) }

class TaskLogic:
    @staticmethod
    def trial_initiation(pattern):
        return { "TrialInitiation": _reader.Harp(f"{pattern}_1", columns=['trial_type']) }

    @staticmethod
    def response(pattern):
        return { "Response": _reader.Harp(f"{pattern}_2", columns=['wall_id', 'poke_id']) }

    @staticmethod
    def pre_trial(pattern):
        return { "PreTrialState": _reader.Harp(f"{pattern}_3", columns=['state']) }

    @staticmethod
    def inter_trial_interval(pattern):
        return { "InterTrialInterval": _reader.Harp(f"{pattern}_4", columns=['state']) }

    @staticmethod
    def slice_onset(pattern):
        return { "SliceOnset": _reader.Harp(f"{pattern}_10", columns=['wall_id']) }

    @staticmethod
    def draw_background(pattern):
        return { "DrawBackground": _reader.Harp(f"{pattern}_11", columns=['state']) }

    @staticmethod
    def gratings_slice_onset(pattern):
        return { "GratingsSliceOnset": _reader.Harp(f"{pattern}_12", columns=['wall_id']) }

class Wall:
    @staticmethod
    def beam_break0(pattern):
        return { "BeamBreak0": _reader.DigitalBitmask(f"{pattern}_32", 0x1, columns=['state']) }

    @staticmethod
    def beam_break1(pattern):
        return { "BeamBreak1": _reader.DigitalBitmask(f"{pattern}_32", 0x2, columns=['state']) }

    @staticmethod
    def beam_break2(pattern):
        return { "BeamBreak2": _reader.DigitalBitmask(f"{pattern}_32", 0x4, columns=['state']) }

    @staticmethod
    def set_led0(pattern):
        return { "SetLed0": _reader.BitmaskEvent(f"{pattern}_34", 0x1, 'Set') }

    @staticmethod
    def set_led1(pattern):
        return { "SetLed1": _reader.BitmaskEvent(f"{pattern}_34", 0x2, 'Set') }

    @staticmethod
    def set_led2(pattern):
        return { "SetLed2": _reader.BitmaskEvent(f"{pattern}_34", 0x4, 'Set') }

    @staticmethod
    def set_valve0(pattern):
        return { "SetValve0": _reader.BitmaskEvent(f"{pattern}_34", 0x8, 'Set') }

    @staticmethod
    def set_valve1(pattern):
        return { "SetValve1": _reader.BitmaskEvent(f"{pattern}_34", 0x10, 'Set') }

    @staticmethod
    def set_valve2(pattern):
        return { "SetValve2": _reader.BitmaskEvent(f"{pattern}_34", 0x20, 'Set') }

    @staticmethod
    def clear_led0(pattern):
        return { "ClearLed0": _reader.BitmaskEvent(f"{pattern}_35", 0x1, 'Clear') }

    @staticmethod
    def clear_led1(pattern):
        return { "ClearLed1": _reader.BitmaskEvent(f"{pattern}_35", 0x2, 'Clear') }

    @staticmethod
    def clear_led2(pattern):
        return { "ClearLed2": _reader.BitmaskEvent(f"{pattern}_35", 0x4, 'Clear') }

    @staticmethod
    def clear_valve0(pattern):
        return { "ClearValve0": _reader.BitmaskEvent(f"{pattern}_35", 0x8, 'Clear') }

    @staticmethod
    def clear_valve1(pattern):
        return { "ClearValve1": _reader.BitmaskEvent(f"{pattern}_35", 0x10, 'Clear') }

    @staticmethod
    def clear_valve2(pattern):
        return { "ClearValve2": _reader.BitmaskEvent(f"{pattern}_35", 0x20, 'Clear') }