
def compositeStream(pattern, *args):
    """Merges multiple data streams into one stream."""
    schema = {}
    if args:
        for stream in args:
            schema.update(stream(pattern))
    return schema

class Device:
    """
    Groups multiple data streams into a logical device.
    
    If a device contains a single stream with the same pattern as the device
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
        self.schema = compositeStream(name, *args)

    def __iter__(self):
        if len(self.schema) == 1:
            singleton = self.schema.get(self.name, None)
            if singleton:
                return iter((self.name, singleton))
        return iter((self.name, self.schema))
