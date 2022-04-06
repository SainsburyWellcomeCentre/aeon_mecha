
def compositeStream(name, *streams):
    """Merges multiple data streams into one stream."""
    schema = {}
    if streams is not None:
        for sch in streams:
            schema.update(sch(name)[1])
    return name, schema

class Device:
    """
    
    Attributes
    ----------
    name : str
        Name of the device.
    schemas : Any
        Data streams collected from the device.
    """
    def __init__(self, name, *schemas):
        self.name = name
        self.schema = compositeStream(name, *schemas)[1]

    def __iter__(self):
        return iter((self.name, self.schema))
