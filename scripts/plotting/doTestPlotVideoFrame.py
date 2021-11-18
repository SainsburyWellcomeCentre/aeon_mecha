import sys
import pdb
import datetime
import plotly.express as px

sys.path.append("../..")
import aeon.preprocess.api
import aeon.preprocess.utils


def main(argv):
    session_index = 31
    mouse_id = "BAA-1099793"
    frame_delay = 6490.0
    video_data_duration_sec = 0.1
    root = "/ceph/aeon/test2/experiment0.1"

    metadata = aeon.preprocess.api.sessiondata(root)
    metadata = metadata[metadata.id.str.startswith('BAA')]
    metadata = aeon.preprocess.utils.getPairedEvents(metadata=metadata)
    metadata = aeon.preprocess.api.sessionduration(metadata)
    mouse_metadata = metadata[metadata.id == mouse_id]
    session_start = mouse_metadata.iloc[session_index]["start"]
    session_end = mouse_metadata.iloc[session_index]["end"]
    frame_start_time = session_start + datetime.timedelta(seconds=frame_delay)
    video_data_end_time = session_start + datetime.timedelta(seconds=frame_delay+video_data_duration_sec)

    video_data = aeon.preprocess.api.videodata(root, 'FrameTop', start=frame_start_time, end=video_data_end_time)
    first_two_video_data_rows = video_data.iloc[0:1]
    frame = next(aeon.preprocess.api.videoframes(first_two_video_data_rows))
    fig = px.imshow(frame, color_continuous_scale="gray")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
