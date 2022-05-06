

def compositeStream(name, *args):
    """Merges multiple data streams into one stream."""
    streams = {}
    if args:
        for stream in args:
            streams.update(stream(name))
    return streams


class Device:
    """
    Groups multiple data streams into a logical device.

    If a device contains a single stream with the same name as the device
    `name`, it will be considered a singleton, and the stream reader will be
    paired directly with the device without nesting.

    Attributes
    ----------
    name : str
        Name of the device.
    args : Any
        Data streams collected from the device.
    """
    def __init__(self, name, *args):
        self.name = name
        self.streams = compositeStream(name, *args)

    def __iter__(self):
        if len(self.streams) == 1:
            singleton = self.streams.get(self.name, None)
            if singleton:
                return iter((self.name, singleton))
        return iter((self.name, self.streams))
