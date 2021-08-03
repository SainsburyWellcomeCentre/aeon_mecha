import sys
import pdb
import numpy as np
import plotly.graph_objects as go
import scipy.interpolate


def main(argv):
    def sigmoidal(x):
        # x \in [0,1]
        # x_scaled \in [-6,6]
        x_scaled = 12*x-6
        answer = 1.0/(1+np.exp(-x_scaled))
        return answer
    wrapper = sigmoidal

    t = np.arange(0, 1.01, .01)
    unew = np.arange(0, 1.1, 0.1)

    x = np.sin(2*np.pi*wrapper(t))
    y = np.cos(2*np.pi*wrapper(t))

    tck, u = scipy.interpolate.splprep([x, y], s=0)
    interpolated = scipy.interpolate.splev(wrapper(unew), tck)

    fig = go.Figure()
    original_data_trace = go.Scatter(x=x, y=y, mode="markers",
                                     customdata=t,
                                     hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                                     name="Linear", showlegend=False)
    fig.add_trace(original_data_trace)

    cubic_interpolation_trace = go.Scatter(x=interpolated[0], y=interpolated[1], mode="markers",
                                           customdata=unew,
                                           hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec<br>",
                                           name="Cubic", showlegend=False)
    fig.add_trace(cubic_interpolation_trace)
    fig.show()
    pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
