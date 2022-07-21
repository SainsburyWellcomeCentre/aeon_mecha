"""
July 2022
Upgrade all timestamps longblob fields with datajoint 0.13.7
"""
import datajoint as dj
from datetime import datetime
import numpy as np
from tqdm import tqdm

assert dj.__version__ >= "0.13.7"


schema = dj.schema("u_thinh_aeonfix")


@schema
class TimestampFix(dj.Manual):
    definition = """
    full_table_name: varchar(64)
    key_hash: uuid  # dj.hash.key_hash(key)    
    """


schema_names = (
    "aeon_acquisition",
    "aeon_tracking",
    "aeon_qc",
    "aeon_report",
    "aeon_analysis",
)


def main():
    for schema_name in schema_names:
        vm = dj.create_virtual_module(schema_name, schema_name)
        table_names = [
            ".".join(
                [
                    dj.utils.to_camel_case(s)
                    for s in tbl_name.strip("`").split("__")
                    if s
                ]
            )
            for tbl_name in vm.schema.list_tables()
        ]
        for table_name in table_names:
            table = get_table(vm, table_name)
            print(f"\n---- {schema_name}.{table_name} ----\n")
            for attr_name, attr in table.heading.attributes.items():
                if "timestamp" in attr_name and attr.type == "longblob":
                    for key in tqdm(table.fetch("KEY")):
                        fix_key = {
                            "full_table_name": table.full_table_name,
                            "key_hash": dj.hash.key_hash(key),
                        }
                        if TimestampFix & fix_key:
                            continue
                        ts = (table & key).fetch1(attr_name)
                        if not len(ts) or isinstance(ts[0], np.datetime64):
                            TimestampFix.insert1(fix_key)
                            continue
                        assert isinstance(ts[0], datetime)
                        with table.connection.transaction:
                            table.update1(
                                {
                                    **key,
                                    attr_name: np.array(ts).astype("datetime64[ns]"),
                                }
                            )
                            TimestampFix.insert1(fix_key)


def get_table(schema_object, table_object_name):
    if "." in table_object_name:
        master_name, part_name = table_object_name.split(".")
        return getattr(getattr(schema_object, master_name), part_name)
    else:
        return getattr(schema_object, table_object_name)


if __name__ == "__main__":
    main()
