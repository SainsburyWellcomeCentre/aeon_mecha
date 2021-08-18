import datajoint as dj

_default_database_prefix = 'aeon_'

# safe-guard in case `custom` is not provided
if 'custom' not in dj.config:
    dj.config['custom'] = {}

db_prefix = dj.config['custom'].get('database.prefix', _default_database_prefix)


def get_schema_name(name):
    return db_prefix + name
