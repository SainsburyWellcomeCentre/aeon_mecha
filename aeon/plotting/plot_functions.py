import pdb
import collections
import plotly.graph_objects as go

def get_trayectory_trace(x, y, time_stamps=None):
    if time_stamps is None:
        trace = go.Scatter(x=x, y=y, mode="markers")
    else:
        trace = go.Scatter(x=x, y=y, mode="markers",
                           marker={"color": time_stamps,
                                   "colorscale": "Rainbow",
                                   "colorbar": {"title": "Time"},
                                  },
                           customdata=time_stamps,
                           hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                           showlegend=False,
                           )
    return trace

def get_patches_traces(patches_coordinates, fill_color="gray", fill_opacity=0.5):
    patches_traces = []
    for i in range(patches_coordinates.shape[0]):
        patch_lower_x, patch_higher_x, patch_lower_y, patch_higher_y = \
            patches_coordinates.loc[i, ["lower_x", "higher_x", 
                                        "lower_y", "higher_y"]]
        patch_xs = [patch_lower_x, patch_higher_x, patch_higher_x, patch_lower_x, patch_lower_x]
        patch_ys = [patch_lower_y, patch_lower_y, patch_higher_y, patch_higher_y, patch_lower_y]
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
