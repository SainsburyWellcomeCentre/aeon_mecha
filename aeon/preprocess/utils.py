import pdb
import numpy as np

def get_positions_labels(positions, patches_coordinates=None,
                         nest_coordinates=None):

    def flag_positions_in_square(positions, square_coordinates):
        samples = ((square_coordinates["lower_x"]<=positions["x"]) &
                   (positions["x"]<square_coordinates["higher_x"]) &
                   (square_coordinates["lower_y"]<=positions["y"]) &
                   (positions["y"]<square_coordinates["higher_y"]))
        samples = samples.to_numpy()
        return samples

    labels = np.array([None]*positions.shape[0])
    if patches_coordinates is not None:
        for i in range(patches_coordinates.shape[0]):
            flags = flag_positions_in_square(positions=positions,
                                            square_coordinates=
                                            patches_coordinates.iloc[i,:])
            labels[flags] = "Patch{:d}".format(i)
    if nest_coordinates is not None:
        flags = flag_positions_in_square(positions=positions,
                                        square_coordinates=
                                        nest_coordinates)
        labels[flags] = "Nest"
    labels[np.equal(labels, None)] = "Exploring"
    return labels
