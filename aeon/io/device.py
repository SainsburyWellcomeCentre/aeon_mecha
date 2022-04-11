
def compositeStream(name, *args):
    """Merges multiple data streams into one stream."""
    schema = {}
    if args:
        for sch in args:
            schema.update(sch(name)[1])
    return name, schema

class Device:
    """
    
    Attributes
    ----------
    name : str
        Name of the device.
    args : Any
        Data streams collected from the device.
    """
    def __init__(self, name, *args):
        self.name = name
        self.schema = compositeStream(name, *args)[1]

    def __iter__(self):
        return iter((self.name, self.schema))
