import pdb
import numpy as np

def getPairedEvents(metadata):
    paired_events = None
    i = 0
    while i < (len(metadata)-1):
        if metadata.iloc[i]["event"]=="Start" and metadata.iloc[i+1]["event"]=="End":
            if paired_events is None:
                paired_events = metadata.iloc[i:(i+2)]
            else:
                paired_events = paired_events.append(metadata.iloc[i:(i+2)])
            i += 2
        else:
            i += 1
    return paired_events

def get_positions_labels(x, y, patches_coordinates=None,
                         nest_coordinates=None):

    def flag_positions_in_square(x, y, square_coordinates):
        samples = ((square_coordinates["lower_x"].item()<=x) &
                   (x<square_coordinates["higher_x"].item()) &
                   (square_coordinates["lower_y"].item()<=y) &
                   (y<square_coordinates["higher_y"].item()))
        return samples

    labels = np.array([None]*len(x))
    if patches_coordinates is not None:
        for i in range(patches_coordinates.shape[0]):
            flags = flag_positions_in_square(x=x, y=y,
                                            square_coordinates=
                                            patches_coordinates.iloc[i,:])
            labels[flags] = "Patch{:d}".format(i+1)
    if nest_coordinates is not None:
        flags = flag_positions_in_square(x=x, y=y,
                                        square_coordinates=
                                        nest_coordinates)
        labels[flags] = "Nest"
    labels[np.equal(labels, None)] = "Other"
    return labels