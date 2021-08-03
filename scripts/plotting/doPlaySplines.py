import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt
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
    plt.scatter(x, y)
    plt.show()

    tck, u = scipy.interpolate.splprep([x, y], s=0)
    out = scipy.interpolate.splev(unew, tck)
    plt.figure()
    plt.plot(x, y, 'x', out[0], out[1], np.sin(2*np.pi*wrapper(unew)), np.cos(2*np.pi*wrapper(unew)), x, y, 'b')
    plt.legend(['Linear', 'Cubic Spline', 'True'])
    plt.axis([-1.05, 1.05, -1.05, 1.05])
    plt.title('Spline of parametrically-defined curve')
    plt.show()

if __name__=="__main__":
    main(sys.argv)
