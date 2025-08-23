import numpy as np


def distancetravelled(angle, radius=4.0):
    '''
    Calculates the total distance travelled on the wheel, by taking into account
    its radius and the total number of turns in both directions across time.

    :param Series angle: A series of magnetic encoder measurements.
    :param float radius: The radius of the wheel, in metric units.
    :return: The total distance travelled on the wheel, in metric units.
    '''
    maxvalue = int(np.iinfo(np.uint16).max >> 2)
    jumpthreshold = maxvalue // 2
    turns = angle.astype(int).diff()
    clickup = (turns < -jumpthreshold).astype(int)
    clickdown = (turns > jumpthreshold).astype(int) * -1
    turns = (clickup + clickdown).cumsum()
    distance = 2 * np.pi * radius * (turns + angle / maxvalue)
    distance = distance - distance[0]
    return distance

def activepatch(wheel, in_patch):
    '''
    Computes a decision boundary for when a patch is active based on wheel movement.
    
    :param Series wheel: A pandas Series containing the cumulative distance travelled on the wheel.
    :param Series in_patch: A Series of type bool containing whether the specified patch may be active.
    :return: A pandas Series specifying for each timepoint whether the patch is active.
    '''
    exit_patch = in_patch.astype(np.int8).diff() < 0
    in_wheel = (wheel.diff().rolling('1s').sum() > 1).reindex(in_patch.index, method='pad')
    epochs = exit_patch.cumsum()
    return in_wheel.groupby(epochs).apply(lambda x:x.cumsum()) > 0
