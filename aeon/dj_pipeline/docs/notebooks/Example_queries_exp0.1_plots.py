# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-toolsai.jupyter added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../../../../../../var/folders/t3/wwync7qs1753cd1nzlqnzt900000gn/T/5c59f61f-ef39-4f82-b9a8-c05ca64ab4b1'))
	print(os.getcwd())
except:
	pass
# %% [markdown]
#   # Using DataJoint for Aeon data
# %% [markdown]
#   ## Imports and Config

# %%
import datetime

import datajoint as dj
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# %% [markdown]
#   ### DataJoint configuration
#
#   Change the settings below to your username and password to connect to `aeon-db` on _hpc-gw1_.
#   Make sure you are connected to either `hpc-gw1.hpc.swc.ucl.ac.uk` or `ssh.swc.ucl.ac.uk`

# %%
# dj.config["database.host"] = "aeon-db"
dj.config["database.user"] = "jburling"
# dj.config["database.password"] = "******"

# %% [markdown]
#  After your configuration has been set up, let DataJoint try to connect to the database.
#  The output from `.list_schemas()` should list read-only schemas starting with `aeon_` in addition to your own user created schemas that have read/write access (if they exists).

# %%
dj.conn()
dj.list_schemas()

# %% [markdown]
#   You can configure additional table display options like so:

# %%
dj.config["display.limit"] = 15  # limit number of displayed rows
dj.config["display.width"] = 25  # limit number of displayed columns

# %% [markdown]
#   Once `config` is setup, you can save your configuration to load automatically with `dj.config.save_local()`.
#   This will save a _json_ file called `dj_local_conf.json` to your current working directory.
#   Here's an example of the `.json` file used for Aeon:
# %% [markdown]
#  ```json
#  {
#      "database.host": "aeon-db",
#      "database.password": "*********",
#      "database.user": "jburling",
#      "database.port": 3306,
#      "database.reconnect": true,
#      "connection.init_function": null,
#      "connection.charset": "",
#      "loglevel": "DEBUG",
#      "safemode": true,
#      "fetch_format": "array",
#      "display.limit": 50,
#      "display.width": 25,
#      "display.show_tuple_count": true,
#      "database.use_tls": null,
#      "enable_python_native_blobs": true,
#      "stores": {
#          "djstore": {
#              "protocol": "file",
#              "location": "/nfs/nhome/live/jburling/djstore",
#              "stage": "/nfs/nhome/live/jburling/djstore"
#          }
#      },
#      "custom": {
#          "database.prefix": "aeon_",
#          "repository_config": {
#              "ceph_aeon": "/ceph/aeon"
#          }
#      }
#  }
#  ```
# %% [markdown]
#  ### Virtual modules vs. import
#
#   You have the option to create _virtual_ modules using `dj.create_virtual_module` or import the schemas directly from the `aeon.dj_pipeline` submodule.
#   Virtual modules can't autopopulate the tables but also don't require the `aeon` package code to access and query the data.
#   If you have access to the latest `dj_pipeline` branch on the `aeon_mecha` repo, and your environment is setup and the `aeon` package installed, you can set `_use_virtual_module=False`.

# %%
_use_virtual_module = True

# %% [markdown]
#   The main data tables on `aeon-db` are saved as schemas starting with `"aeon_"`. We'll use this prefix to create the virtual modules.

# %%
_db_prefix = "aeon_"

if _use_virtual_module:
    acquisition = dj.create_virtual_module("acquisition", _db_prefix + "acquisition")
    analysis = dj.create_virtual_module("analysis", _db_prefix + "analysis")
    lab = dj.create_virtual_module("lab", _db_prefix + "lab")
    subject = dj.create_virtual_module("subject", _db_prefix + "subject")
    tracking = dj.create_virtual_module("tracking", _db_prefix + "tracking")

else:
    dj.config.update(custom={**dj.config.get("custom", {}), "database.prefix": _db_prefix})
    from aeon.dj_pipeline import acquisition, analysis, lab, subject, tracking

# %% [markdown]
#  ## Helper functions
#
#  Helper functions used later for concatenating position tracking slices.

# %%
def check_fetch_len(key, length=1):
    """
    Check that a key is of a certain length

    :param key: A key must be a list, query, or pandas DF (not a dict)
    :type key: list, QueryExpression, DataFrame
    :param length: Length to use to check key, defaults to 1
    :type length: int, optional
    :raises ValueError: Key is incorrect length
    """
    assert isinstance(key, (list, dj.expression.QueryExpression, pd.DataFrame))
    if not len(key) == length:
        raise ValueError(f"Key must be of length {length}")


def position_concat(session_key, acquisition, tracking, pixel_scale=0.00192):
    """
    Concatenate position data into a single pandas DataFrame

    :param session_key: a key for a single session
    :type session_key: [type]
    :param acquisition: DataJoint module
    :type acquisition: dj.Schema
    :param tracking: DataJoint module
    :type tracking: dj.Schema
    :param pixel_scale: convert pixels to mm, defaults to 0.00192
    :type pixel_scale: float, optional
    :return: A DataFrame representation of the table
    :rtype: pd.DataFrame
    """
    sess_key = (acquisition.Session() & session_key).fetch(as_dict=True)

    check_fetch_len(sess_key, 1)
    sess_key = sess_key[0]

    to_expand = [
        "timestamps",
        "position_x",
        "position_y",
        "position_z",
        "area",
        "speed",
    ]

    pos_arr = (tracking.SubjectPosition() & sess_key).fetch(
        *to_expand, order_by="time_slice_start"
    )

    for idx, field in enumerate(to_expand):
        col_data = pos_arr[idx]
        col_data = np.concatenate(col_data)
        if field != "timestamps":
            col_data *= pixel_scale
        sess_key[field] = col_data

    return pd.DataFrame(sess_key).set_index("timestamps")

# %% [markdown]
#   ## Relevant tables
#   The following tables are important for querying data needed for plotting and general summary statistics.
#
#  _Note:_ to view a subset of a tables' contents, call it like a function, e.g., `acquisition.Session()` and not `acquisition.Session`.

# %%
# basic info about a session
acquisition.Session()


# %%
# similar to Session but with `session_end` and `session_duration`, if available.
acquisition.SessionEnd()


# %%
# table for position tracking data
tracking.SubjectPosition()

# %% [markdown]
#  ## Basic DataJoint queries and finding a good session
#
#  The `*` operator performs an inner join. See this link for more details: <https://docs.datajoint.org/python/v0.13/queries/07-Join.html#join-operator>
#
#  Here we'll combine `session_end` fields with the Session table, keeping only matching records.

# %%
acquisition.Session * acquisition.SessionEnd

# %% [markdown]
#  Here we'll further restrict the joined table to subset based on some duration condition.
#  In this case, sessions which last longer than 4 hours.

# %%
(acquisition.Session * acquisition.SessionEnd) & "session_duration > 4"

# %% [markdown]
#  Well define sessions with end times and long durations as `good_sessions`, and use that query table to fetch some data from the database.

# %%
good_sessions = (acquisition.Session * acquisition.SessionEnd) & "session_duration > 4"
subject, session_duration = good_sessions.fetch("subject", "session_duration")
session_duration[:5]  # print first five only

# %% [markdown]
#  If you use the special string `"KEY"`, then you can fetch only the primary keys from the table.

# %%
good_session_keys = good_sessions.fetch("KEY")
good_session_keys

# %% [markdown]
#  From the list of good session primary keys, select a single session to use for further queries. Here we'll just take the first one.

# %%
session_key = good_session_keys[0]

# %% [markdown]
#  The object `session_key` is a dictionary of primary keys and their values to be used to select a valid session from different tables.

# %%
session_key

# %% [markdown]
# ### Alternative methods to find a single session
#
# Find a session for given a subject ID and a star time criterion.

# %%
session_conditional = (
    good_sessions & 'subject = "BAA-1099790"' & 'session_start > "2021-08-01"'
)

session_key = session_conditional.fetch("KEY")[0]
session_conditional

# %% [markdown]
# Find a session based on the one with the most time slices
#
# 1. In `SubjectPosition`, group by all the unique combinations of primary keys found in `good_sessions`, then count the number of `time_slice_start` entries for each group
# 2. Get the max number of time slice counts
# 3. Use the max stored in `max_count` to filter `time_counts` and get the last primary key found and use it for `session_key`

# %%
time_counts = good_sessions.aggr(tracking.SubjectPosition, n_obs="count(time_slice_start)")
max_count = time_counts.fetch("n_obs").max()
session_key = (time_counts & {"n_obs": max_count}).fetch("KEY")[-1:]
time_counts

# %% [markdown]
#  We can now use the single session as key to do restrictions/subsetting on other DataJoint tables.
#
#  Subsetting position data for a single session:

# %%
tracking.SubjectPosition & session_key

# %% [markdown]
#  ### Additional tables of interest
#
#  See the diagram from `aeon/dj_pipeline/docs/notebooks/diagram.svg` for more tables.

# %%
# summary statistics for one session
analysis.SessionSummary() & session_key


# %%
# summary statistics for food pathces for a single session
analysis.SessionSummary.FoodPatch & session_key


# %%
# summary statistics for time in arena or corridor for a session
analysis.SessionTimeDistribution & session_key


# %%
# summary statistics for time in food patch for a session
analysis.SessionTimeDistribution.FoodPatch & session_key


# %%
# summary statistics for time in nest for a session
analysis.SessionTimeDistribution.Nest & session_key

# %% [markdown]
#  Using these tables together in practice

# %%
# join Session timestamp tables then subset using `session_key`
session_times = (acquisition.Session * acquisition.SessionEnd) & session_key

# extract session time stamps from the above query table
start_time, end_time = session_times.fetch1("session_start", "session_end")

# string to further restrict patch events to a specific time range
time_range_str = f"event_time BETWEEN '{start_time}' AND '{end_time}'"
patch_events = acquisition.FoodPatchEvent & time_range_str
patch_events


# %%
# lookup table mapping codes to event types
acquisition.EventType()


# %%
# joint `patch_events` data with EventType to combine with event types and event codes
acquisition.EventType * patch_events

# %% [markdown]
#   ## Generating the traceplot
#
#   Use the `session_key` to subset data from `SubjectPosition` and concatenating all individual time slices into a single data frame.
#   A similar concatenation function is also found in the `aeon` package under `aeon.dj_pipeline.tracking` and called `SubjectPosition.get_session_position()`

# %%
pos_data = position_concat(session_key, acquisition, tracking, pixel_scale=1)
pos_data = pos_data[:50_000]  # subsetting for performance reasons
pos_data

# %% [markdown]
#   ### Additional plotting helpers taken from `joacorapela`

# %%
def get_trajectory_trace(x, y, time_stamps):
    trace = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker={
            "color": time_stamps,
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
        patch_lower_x, patch_higher_x, patch_lower_y, patch_higher_y = patches_coordinates.loc[
            i, ["lower_x", "higher_x", "lower_y", "higher_y"]
        ]
        patch_xs = [
            patch_lower_x,
            patch_higher_x,
            patch_higher_x,
            patch_lower_x,
            patch_lower_x,
        ]
        patch_ys = [
            patch_lower_y,
            patch_lower_y,
            patch_higher_y,
            patch_higher_y,
            patch_lower_y,
        ]
        patch_trace = go.Scatter(
            x=patch_xs,
            y=patch_ys,
            fill="toself",
            fillcolor=fill_color,
            opacity=fill_opacity,
            mode="none",
            showlegend=False,
        )
        patches_traces.append(patch_trace)
    return patches_traces


# %%
x = pos_data["position_x"].to_numpy()
y = pos_data["position_y"].to_numpy()
session_start = pos_data.session_start[0].to_numpy()
time_stamps = (pos_data.index - session_start).total_seconds().to_numpy()
duration_sec = (acquisition.SessionEnd & session_key).fetch1("session_duration") * 3600

title = "Start {:s}, (max={:.02f} sec)".format(str(session_start), duration_sec)

patches_coordinates = pd.DataFrame(
    data=np.matrix("584,597,815,834;614,634,252,271"),
    columns=["lower_x", "higher_x", "lower_y", "higher_y"],
)

nest_coordinates = pd.DataFrame(
    data=np.matrix("170,260,450,540"), columns=["lower_x", "higher_x", "lower_y", "higher_y"]
)

fig = go.Figure()

trace = get_trajectory_trace(x, y, time_stamps)

fig.add_trace(trace)

patches_traces = get_patches_traces(patches_coordinates=patches_coordinates)

for patch_trace in patches_traces:
    fig.add_trace(patch_trace)

nest_trace = get_patches_traces(patches_coordinates=nest_coordinates)[0]

fig.add_trace(nest_trace)

fig.update_layout(
    title=title,
    xaxis_title="x (pixels)",
    yaxis_title="y (pixels)",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    width=700,
    height=700,
    margin=dict(l=50, r=50, b=100, t=100, pad=4),
)

# %% [markdown]
#   # Appendix
#   ## `aeon` package setup and requirements
#
#   TODO
# %% [markdown]
#  ## Additional helper functions
#
#  The functions below help to find complete sessions using information stored among different tables.

# %%
def join_timestamps(acquisition, keep_null=False):
    # get subject keys, session start & end, time_slice start & end
    return acquisition.Session.join(acquisition.SessionEnd, left=keep_null).join(
        acquisition.TimeSlice, left=keep_null
    )


def keys_complete_sessions(acquisition):
    all_timestamps = join_timestamps(acquisition, keep_null=True)

    # session_start time but no chunk_start time
    nonstart_sessions = acquisition.Session & (all_timestamps & "ISNULL(chunk_start)")

    # session_start time but no session_end time
    ongoing_sessions = acquisition.Session & (all_timestamps & "ISNULL(session_end)")

    return acquisition.Session - nonstart_sessions - ongoing_sessions


def keys_complete_tracking(acquisition, tracking):
    sessions = keys_complete_sessions(acquisition)
    tracking_data = sessions.join(tracking.SubjectPosition, left=True)

    incomplete_tracking_sessions = acquisition.Session & (tracking_data & "ISNULL(timestamps)")

    return sessions - incomplete_tracking_sessions
