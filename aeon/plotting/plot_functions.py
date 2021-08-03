import collections
import numpy as np
import scipy.interpolate
import plotly.graph_objects as go


def get_trayectory_trace(x, y, time_stamps=None, sample_rate=None,
                         colorscale="Rainbow", opacity=0.3):
    if time_stamps is None:
        trace = go.Scatter(x=x, y=y, mode="markers")
    else:
        if sample_rate is not None:
            min_time = time_stamps.min()
            max_time = time_stamps.max()
            dt = 1.0/sample_rate
            # we need to remove nan from x and y before calling splprep
            nan_indices = set(np.where(np.isnan(x))[0])
            nan_indices.update(np.where(np.isnan(y))[0])
            nan_indices_list = sorted(nan_indices)
            x = np.delete(x, nan_indices_list)
            y = np.delete(y, nan_indices_list)
            time_stamps = np.delete(time_stamps, nan_indices_list)
            # done removing nan from x and y
            tck, u = scipy.interpolate.splprep([x, y], s=0, u=time_stamps)
            time_stamps = np.arange(min_time, max_time, dt)
            x, y = scipy.interpolate.splev(time_stamps, tck)
        trace = go.Scatter(x=x, y=y, mode="markers",
                           marker={"color": time_stamps,
                                   "opacity": opacity,
                                   "colorscale": colorscale,
                                   "colorbar": {"title": "Time"}},
                           customdata=time_stamps,
                           hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                           showlegend=False,
                           )
    return trace


def get_patches_traces(patches_coordinates, fill_color="gray", 
                       fill_opacity=0.5):
    patches_traces = []
    for i in range(patches_coordinates.shape[0]):
        patch_lower_x, patch_higher_x, patch_lower_y, patch_higher_y = \
            patches_coordinates.loc[i, ["lower_x", "higher_x",
                                        "lower_y", "higher_y"]]
        patch_xs = [patch_lower_x, patch_higher_x, patch_higher_x,
                    patch_lower_x, patch_lower_x]
        patch_ys = [patch_lower_y, patch_lower_y, patch_higher_y,
                    patch_higher_y, patch_lower_y]
        patch_trace = go.Scatter(x=patch_xs, y=patch_ys, fill="toself",
                                 fillcolor=fill_color, opacity=fill_opacity,
                                 mode="none", showlegend=False)
        patches_traces.append(patch_trace)
    return patches_traces


def get_cumTimePerActivity_barplot_trace(positions_labels):
    counter = collections.Counter(positions_labels.tolist())
    x = list(counter.keys())
    x.sort()
    y = [counter[x[i]]/len(positions_labels) for i in range(len(x))]
    trace = go.Bar(x=x, y=y)
    return trace


def get_travelled_distance_trace(travelled_seconds, travelled_distance,
                                 color="blue", showlegend=False):
    trace = go.Scatter(x=travelled_seconds, y=travelled_distance,
                       line=dict(color=color), showlegend=showlegend)
    return trace
