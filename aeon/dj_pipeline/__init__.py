import datajoint as dj

_default_database_prefix = 'aeon_'

dj.config['display.width'] = 30

# safe-guard in case `custom` is not provided
if 'custom' not in dj.config:
    dj.config['custom'] = {}


def get_schema_name(name):
    prefix = dj.config['custom'].get('database.prefix', _default_database_prefix)
    return prefix + name
