import pdb
import numpy as np

def get_positions_labels(x, y, patches_coordinates=None,
                         nest_coordinates=None):

    def flag_positions_in_square(x, y, square_coordinates):
        samples = ((square_coordinates["lower_x"]<=x) &
                   (x<square_coordinates["higher_x"]) &
                   (square_coordinates["lower_y"]<=y) &
                   (y<square_coordinates["higher_y"]))
        return samples

    labels = np.array([None]*len(x))
    if patches_coordinates is not None:
        for i in range(patches_coordinates.shape[0]):
            flags = flag_positions_in_square(x=x, y=y,
                                            square_coordinates=
                                            patches_coordinates.iloc[i,:])
            labels[flags] = "Patch{:d}".format(i)
    if nest_coordinates is not None:
        flags = flag_positions_in_square(x=x, y=y,
                                        square_coordinates=
                                        nest_coordinates)
        labels[flags] = "Nest"
    labels[np.equal(labels, None)] = "Exploring"
    return labels
