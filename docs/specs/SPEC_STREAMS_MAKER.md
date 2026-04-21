# Streams Maker Architecture

Auto-generates DataJoint table definitions for device and stream data based on Pydantic schema definitions.

## Overview

`streams_maker.py` bridges declarative Pydantic schema definitions (Experiment classes with Rig objects) and executable DataJoint tables (`streams.py`). It reads catalog entries from the database (`StreamType`, `DeviceType`, `DeviceName`, `Device`) and generates Python table classes dynamically.

## Package Dependencies

```
swc.aeon.schema (aeon_api)
├── BaseSchema
├── data_reader decorator
├── Device, SpinnakerCamera, UndergroundFeeder
└── Reader classes (Video, Position, Encoder)
        │
        ▼
Experiment packages (e.g., aeon_exp_foragingABC)
├── Extend device classes with @data_reader methods
└── Define experiment-specific Rig(BaseSchema)
```

**Key imports** (in experiment package):
```python
from swc.aeon.schema import BaseSchema, data_reader
from swc.aeon.schema.core import Video, Position, Encoder
from swc.aeon.schema.video import SpinnakerCamera
from swc.aeon.schema.foraging import UndergroundFeeder
from swc.aeon.io import reader
```

## Three Decoupled Steps

The streams_maker architecture is organized into three distinct steps that **must execute in order** but are **decoupled in their dependencies**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         THREE DECOUPLED STEPS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: CATALOG POPULATION (Class-level)                                   │
│  ─────────────────────────────────────────                                  │
│  Input:  Pydantic Experiment class                                          │
│  Output: DeviceType, StreamType, DeviceType.Stream tables                   │
│  When:   Experiment registration / worker startup                           │
│  Needs:  Pydantic class definition only (NO metadata.json)                  │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 2: TABLE CREATION (DDL)                                               │
│  ────────────────────────────                                               │
│  Input:  DeviceType, StreamType catalog                                     │
│  Output: ExperimentDevice, DeviceDataStream table classes                   │
│  When:   Worker startup (MUST be outside transaction)                       │
│  Needs:  Catalog only (columns extracted on-demand from reader class)       │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 3: DATA POPULATION (DML)                                              │
│  ─────────────────────────────                                              │
│  Input:  metadata.json (instances), data files (stream data)                │
│  Output: Entries in Device, ExperimentDevice, DeviceDataStream tables       │
│  When:   EpochConfig.populate(), DeviceDataStream.populate()                │
│  Needs:  metadata.json for device instances, EpochConfig.Meta for readers   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Insight: Columns are Class-Level Information

**Stream columns are defined by the reader CLASS, not by runtime/epoch-specific configuration.**

```python
class Video(reader.Harp):
    def __init__(self, pattern):
        # Columns are FIXED in the class definition!
        super().__init__(pattern, columns=["hw_counter", "hw_timestamp"])
```

The `pattern` argument affects WHERE data is loaded from, not WHAT columns exist. This means:
- **Step 2** can extract columns on-demand by importing the reader class from `StreamType.stream_reader`
- **No need to store columns** in the catalog - they're always available from the reader class
- **Step 3** uses metadata.json only for device INSTANCES (rows), not table SCHEMA (columns)

### Why This Separation Matters

MySQL forbids DDL (`CREATE TABLE`) inside transactions. Since `dj.Imported.populate()` wraps `make()` in a transaction, table creation must happen at worker startup (outside transactions), not during `EpochConfig.make()`.

Columns are extracted on-demand by importing the reader class directly from `StreamType.stream_reader`, avoiding any dependency on `EpochConfig.Meta` during table creation.

## Architecture Flow

```
┌─────────────────────────────────┐
│ Experiment Class                │
│ (Pydantic BaseSchema)           │
│ • "module.path:Experiment"      │
│ • Experiment.rig → Rig          │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ load_metadata.py                │
│                                 │
│ STEP 1 (Catalog - class-level): │
│ • get_experiment_pydantic()     │ ← Loads Experiment class from "module:Class" path
│ • populate_catalog_from_pydantic()│ ← Populates catalog from class definition
│                                 │
│ STEP 3 (DML - instance-level):  │
│ • insert_device_types()         │ ← Populates Device table (serial numbers)
│ • ingest_epoch_metadata_from_rig()│ ← Inserts device installations
│ • get_stream_reader_for_epoch() │ ← Runtime reader resolution (for data loading)
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Database Tables                 │
│                                 │
│ Catalog (Step 1):               │
│ • StreamType                    │ ← Stores stream_reader path for class import
│ • DeviceType                    │
│ • DeviceType.Stream             │
│                                 │
│ Instance Data (Step 3):         │
│ • Device                        │ ← Serial numbers from metadata.json
│ • EpochConfig.Meta (json)       │ ← Stores rig_config for reader resolution
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ streams_maker.main()  (STEP 2)  │
│ • get_device_template()         │
│ • get_device_stream_template()  │ ← Imports reader class, extracts columns on-demand
│                                 │
│ ⚠️  MUST run outside transaction │
│ ⚠️  Called at worker startup     │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ streams.py (auto-generated)     │
│ • Device tables (dj.Manual)     │
│ • Stream tables (dj.Imported)   │
└─────────────────────────────────┘
```

## Key Concepts

### Rig

A Pydantic model representing the hardware configuration of an experiment. Extends `BaseSchema` which provides `_validate_container_prefix` for `@data_reader` pattern resolution. Contains device collections organized by category:

```python
class Rig(BaseSchema):  # BaseSchema provides _validate_container_prefix for @data_reader
    cameras: Dict[CameraName, Camera]   # e.g., 13 cameras keyed by enum
    feeders: Dict[FeederName, Feeder]   # e.g., 6 feeders keyed by enum
    nest: Dict[NestName, WeightScale]   # Weight scale(s)
```

### Device

A physical or logical hardware unit. A class is identified as a device if it has `@data_reader` decorated methods. Each device:
- Is identified by its **class name** (e.g., `Camera.__name__` → `"Camera"`, `Feeder.__name__` → `"Feeder"`)
- May have a `serial_number` or `port_name` for identification
- Contains `@data_reader` methods that define its data streams

> **Note:** The class name (`cls.__name__`) is used as `device_type` in the database, not the inherited `device_type` Literal field. This avoids parent class names propagating through inheritance (e.g., `Feeder(UndergroundFeeder(HarpOutputExpander))` → `"Feeder"` not `"HarpOutputExpander"`).

### Stream

A data collection channel from a device, defined as a `@data_reader` method on the Device class:

```python
class Camera(SpinnakerCamera):
    @data_reader
    def video(self, pattern) -> reader.Video:
        """Video stream from camera."""
        return Video(f"{pattern}").reader  # Note: returns .reader attribute

    @data_reader
    def position(self, pattern) -> reader.Position:
        """Position tracking stream."""
        return Position(f"{pattern}").reader
```

The `@data_reader` decorator:
- Creates a cached property on the device instance
- Resolves file patterns using `_resolve_pattern_prefix()` based on device hierarchy
- Returns a **reader instance** (via `.reader` attribute) configured for that device's data location

### Pattern Resolution

Patterns in `@data_reader` methods are resolved relative to the device's position in the Rig hierarchy:

```python
# In Rig.cameras["top"].video(pattern="*.mp4")
# Pattern resolves to: <experiment_root>/cameras/top/*.mp4
```

This allows devices to reference their data files without hardcoding paths.

## Key Components

### Catalog Tables

**`StreamType`**: Catalog of unique stream types across all experiments
```python
definition = """
stream_hash: uuid  # hash(stream_type, stream_reader) - unique identifier
---
stream_type: varchar(36)  # e.g., "Video", "WeightRaw"
stream_reader: varchar(256)  # e.g., "swc.aeon.io.reader.Video"
stream_reader_kwargs=null: longblob  # JSON dict of reader constructor kwargs (value, tag, columns, etc.)
stream_description='': varchar(256)
unique index(stream_type, stream_reader)
"""
```

The `stream_hash` serves as the primary key because different experiments may use the same `stream_type` name with different reader implementations. The hash uniquely identifies each (stream_type, stream_reader) combination.

The `stream_reader` field stores the fully-qualified class path (e.g., `"swc.aeon.io.reader.Video"`), which allows Step 2 to import the reader class directly and extract columns on-demand.

**`DeviceType`**: Catalog of device types
- `device_type`: Leaf class name (e.g., `"Camera"`, `"Feeder"`, `"ActivityWeightScale"`)
- `device_description`: Optional description

**`DeviceType.Stream`**: Links device types to their streams
```python
definition = """
-> master
-> StreamType  # References StreamType by stream_hash
"""
```

**`DeviceName`**: Catalog of device instance names
- `device_name`: Instance name (e.g., `"CameraTop"`, `"Feeder1"`)
- Foreign key to `DeviceType`

**`Device`**: Physical device instances
- `device_serial_number`: Unique identifier (or port_name)
- Foreign key to `DeviceType`

### Metadata Storage

**`EpochConfig.Meta`**: Stores epoch metadata including original rig configuration
```python
definition = """
-> master
---
bonsai_workflow: varchar(36)
commit: varchar(64)
source='': varchar(16)
metadata: json  # Original rig_config JSON for Pydantic reconstruction
metadata_file_path: varchar(255)
"""
```

The `metadata` field stores the **original nested rig configuration** as JSON, enabling Pydantic Rig reconstruction from the database without file I/O. This is used **only for runtime stream reader resolution** (Step 3), NOT for table creation (Step 2).

### Parsing Functions

**`get_experiment_pydantic(schema_name)`**
- Dynamically imports Experiment class from module path
- Uses colon separator format: `"module.path:ClassName"`
- Example: `"swc.aeon.exp.foragingABC.experiment:Experiment"` → Experiment class

**`_has_data_readers(cls)`**
- Checks if a class is a "device" by testing for `@data_reader` decorated methods
- Returns `True` if `get_data_reader_methods(cls)` is non-empty
- Used throughout the module to replace the deprecated `hasattr(x, "device_type")` checks

**`get_device_class_from_field(field_info)`**
- Extracts Device class from Pydantic field annotation
- Handles both `Dict[Name, Device]` and single Device field types
- Identifies devices using `_has_data_readers()` (presence of `@data_reader` methods)

**`populate_catalog_from_pydantic(experiment_class)`** *(NEW)*
- **Step 1 function**: Populates catalog from Pydantic class definition
- Extracts DeviceType, StreamType, DeviceType.Stream from class hierarchy
- Does NOT require metadata.json or any epoch data
- Idempotent: safe to call multiple times
- Replaces/improves upon `insert_stream_types()` and `insert_device_types()` for catalog population

```python
def populate_catalog_from_pydantic(experiment_class):
    """Populate catalog tables from Pydantic Experiment class (Step 1).

    Extracts DeviceType, StreamType, and their relationships from the class
    hierarchy. Does NOT extract or store columns - columns are extracted
    on-demand in Step 2 by importing the reader class.

    Args:
        experiment_class: Pydantic Experiment class with rig field
    """
    rig_class = experiment_class.model_fields["rig"].annotation

    for field_name, field_info in rig_class.model_fields.items():
        device_class = get_device_class_from_field(field_info)
        device_type = device_class.__name__  # Leaf class name, e.g., "Camera", "Feeder"

        # Insert DeviceType
        DeviceType.insert1({"device_type": device_type}, skip_duplicates=True)

        # Extract @data_reader methods → StreamType (stream_reader path for later import)
        for stream_name, stream_method in get_data_reader_methods(device_class):
            stream_reader_path = get_reader_path_from_annotation(stream_method)
            stream_hash = compute_hash(stream_name, stream_reader_path)

            StreamType.insert1({
                "stream_hash": stream_hash,
                "stream_type": to_pascal_case(stream_name),
                "stream_reader": stream_reader_path,  # Used in Step 2 to import reader class
            }, skip_duplicates=True)

            DeviceType.Stream.insert1({
                "device_type": device_type,
                "stream_hash": stream_hash,
            }, skip_duplicates=True)
```

**`get_device_info(rig)`**
- Iterates over Rig model fields to find device collections
- Identifies devices by checking for `@data_reader` methods via `_has_data_readers()`
- For each device, extracts:
  - `device_type` from `type(device).__name__` (leaf class name)
  - Stream types from `@data_reader` methods on the device class
- Returns dict mapping device names to their stream info (flat lists):
```python
{
    "CameraTop": {
        "stream_type": ["Video", "Position"],
        "stream_reader": ["swc.aeon.io.reader.Video", "swc.aeon.io.reader.Harp"],
        "stream_hash": [UUID(...), UUID(...)]
    },
    "CameraSide": {...},
    "Feeder1": {...}
}
```

**`get_device_mapper_from_rig(rig)`**
- Extracts device type and serial number mappings
- Uses `type(device).__name__` for device type (leaf class name)
- Identifies devices via `_has_data_readers()` (presence of `@data_reader` methods)
- Handles both dict collections and single device instances

**`insert_stream_types(rig)`** *(superseded by `populate_catalog_from_pydantic()`)*
- Populates `StreamType` table with unique (stream_type, stream_reader) combinations
- Computes `stream_hash` for each entry
- Stores `stream_reader` path for on-demand column extraction in Step 2

**`insert_device_types(rig, metadata_filepath)`** *(superseded by `populate_catalog_from_pydantic()`)*
- Populates catalog tables (`DeviceType`, `DeviceType.Stream`, `Device`)
- Only inserts devices that exist in both Rig and metadata file
- Handles FK constraint by calling `insert_stream_types()` if needed

**`get_stream_reader_for_epoch(experiment_name, device_name, stream_type, epoch_start)`**
- **Runtime stream reader resolution** - used in Step 3 for data loading
- Process:
  1. Get Experiment class path from `Experiment.DevicesSchema` (e.g., `"swc...experiment:Experiment"`)
  2. Fetch `rig_config` JSON from `EpochConfig.Meta`
  3. Load Experiment class using `get_experiment_pydantic()`
  4. Reconstruct Rig using `model_validate()` on the rig config
  5. Find device by name in Rig hierarchy using `_find_device_in_rig()`
  6. Call `@data_reader` method to get configured stream reader
- Returns reader instance ready for `io_api.load()`
- **Note**: This is only used for DATA LOADING (Step 3), NOT for column extraction (Step 1)

**`ingest_epoch_metadata_from_rig(experiment_name, rig, epoch_config, metadata_filepath)`**
- Inserts device installation/removal records
- Handles device attributes (settings/configurations)
- Tracks device removal times

### Template Generators

**`get_device_template(device_type)`**
- Creates `dj.Manual` table (ExperimentDevice) for device installation/removal tracking
- Uses `-> DeviceName` as primary key (not `-> Device`), making queries more intuitive
- Includes `Attribute` and `RemovalTime` part tables
- Example: `Camera` table tracks when cameras are installed/removed

**`get_device_stream_template(device_type, stream_type, streams_module)`**
- Creates `dj.Imported` table for raw data streams
- **Column extraction** (on-demand):
  1. Query `StreamType` catalog for `stream_reader` path
  2. Import the reader class directly
  3. Instantiate with dummy pattern to get columns (columns are instance attributes)
  4. No `EpochConfig.Meta` query needed!
- **make() method**: Uses `get_stream_reader_for_epoch()` for data loading
- Example: `CameraVideo` table stores video metadata per chunk

```python
def get_device_stream_template(device_type, stream_type, streams_module):
    """Create DeviceDataStream table class.

    Columns are extracted on-demand by importing the reader class from
    the stream_reader path stored in StreamType. No EpochConfig.Meta needed.
    """
    # Get stream_reader path from catalog
    stream_detail = (
        DeviceType.Stream * StreamType
        & {"device_type": device_type, "stream_type": stream_type}
    ).fetch1()

    stream_reader_path = stream_detail["stream_reader"]  # e.g., "swc.aeon.io.reader.BitmaskEvent"
    stream_reader_kwargs = stream_detail.get("stream_reader_kwargs") or {}

    # Import the reader class directly
    module_path, class_name = stream_reader_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    reader_class = getattr(module, class_name)

    # Instantiate with dummy pattern and stored kwargs to get columns
    # (columns are instance attributes set in __init__, not class attributes)
    reader_instance = reader_class("_dummy_pattern_", **stream_reader_kwargs)
    columns = reader_instance.columns

    # Filter and normalize columns
    table_columns = []
    for col in columns:
        if col.startswith("_"):
            continue  # Skip metadata columns
        col_name = re.sub(r"\([^)]*\)", "", col).strip()  # Remove type annotations
        table_columns.append(f"{col_name}: longblob")

    # Generate table definition (ExperimentDevice is referenced via device_type name)
    table_definition = f"""
    -> {device_type}          # References ExperimentDevice (e.g., -> Camera)
    -> acquisition.Chunk
    ---
    sample_count: int
    timestamps: longblob
    {chr(10).join(table_columns)}
    """

    # ... create and return table class
```

## Column Extraction Process

```mermaid
graph TD
    A[Worker Startup] --> B[For each registered Experiment]
    B --> C[populate_catalog_from_pydantic]
    C --> D[Get Rig class from Experiment]
    D --> E[For each device field in Rig]
    E --> F[Use cls.__name__ as device_type]
    F --> G[Insert DeviceType]
    G --> H[For each @data_reader method]
    H --> I[Get return type annotation]
    I --> J[Insert StreamType with stream_reader path]
    J --> K[Insert DeviceType.Stream]
    K --> L[streams_maker.main]
    L --> M[get_device_stream_template]
    M --> N[Query StreamType for stream_reader path]
    N --> O[Import reader class]
    O --> P[Instantiate with dummy pattern]
    P --> Q[Extract reader.columns]
    Q --> R[Generate table definition]
    R --> S[Create table class]
```

**Process**:
1. At worker startup, `populate_catalog_from_pydantic()` is called for each registered Experiment
2. For each device type in the Rig class:
   - Insert into `DeviceType` table
   - For each `@data_reader` method:
     - Get return type annotation (e.g., `reader.BitmaskEvent`)
     - Insert into `StreamType` with `stream_reader` path (e.g., `"swc.aeon.io.reader.BitmaskEvent"`)
     - Insert into `DeviceType.Stream`
3. `streams_maker.main()` creates table classes:
   - Query `StreamType` for `stream_reader` path
   - Import the reader class directly
   - Instantiate with dummy pattern to get columns
   - Generate table definition
   - Create and register table class

**Example**:
```python
# Reader class definition (in aeon_api):
class BitmaskEvent(Harp):
    def __init__(self, pattern):
        super().__init__(pattern, columns=["event"])

# Step 1: Store stream_reader path in StreamType (no columns stored)
StreamType.insert1({
    "stream_hash": ...,
    "stream_type": "BeamBreak",
    "stream_reader": "swc.aeon.io.reader.BitmaskEvent",  # Path for import
})

# Step 2: Extract columns on-demand (no EpochConfig.Meta needed!)
stream_reader_path = (StreamType & {"stream_type": "BeamBreak"}).fetch1("stream_reader")
# Returns: "swc.aeon.io.reader.BitmaskEvent"

# Import and instantiate to get columns
module = importlib.import_module("swc.aeon.io.reader")
reader_class = getattr(module, "BitmaskEvent")
columns = reader_class("_dummy_").columns
# Returns: ("event",)
```

## Stream Reader Resolution at Runtime

The `make()` method in generated stream tables uses `get_stream_reader_for_epoch()` to resolve readers for **data loading** (Step 3):

```python
def make(self, key):
    """Load and insert data for DeviceDataStream table."""
    from swc.aeon.io import api as io_api
    from aeon.dj_pipeline.utils.load_metadata import get_stream_reader_for_epoch

    # Fetch chunk info including epoch_start (epoch_start is FK attribute, not in key)
    chunk_start, chunk_end, epoch_start = (acquisition.Chunk & key).fetch1(
        "chunk_start", "chunk_end", "epoch_start"
    )
    data_dirs = acquisition.Experiment.get_data_directories(key)

    # device_name is part of the key (PK via DeviceName), no need to fetch
    device_name = key["device_name"]

    # Get stream reader using Pydantic approach (reconstructs Rig from stored metadata)
    # This is Step 3: we need EpochConfig.Meta for proper pattern resolution
    stream_reader = get_stream_reader_for_epoch(
        key["experiment_name"], device_name, "{stream_type}", epoch_start
    )

    stream_data = io_api.load(
        root=data_dirs,
        reader=stream_reader,
        start=pd.Timestamp(chunk_start),
        end=pd.Timestamp(chunk_end),
    )

    self.insert1({
        **key,
        "sample_count": len(stream_data),
        "timestamps": stream_data.index.values,
        **{col: stream_data[col].values for col in stream_reader.columns if not col.startswith("_")},
    })
```

**Key benefits of this approach**:
1. **No file I/O**: Metadata is fetched from database, not re-read from file
2. **Epoch-specific**: Each epoch can have different device configurations
3. **Pydantic validation**: Rig reconstruction uses `model_validate()` for type safety
4. **Pattern resolution**: `@data_reader` methods automatically resolve file patterns based on device hierarchy

**Note**: `get_stream_reader_for_epoch()` queries `EpochConfig.Meta`, but this is only used for **data loading** (Step 3), NOT for **table creation** (Step 2). By the time `make()` is called, the tables already exist.

## Stream Name Conversion

Stream names are converted from snake_case (method names) to PascalCase (catalog entries):

- `video` → `Video`
- `weight_raw` → `WeightRaw`
- `beam_break` → `BeamBreak`

This conversion is handled by `to_pascal_case()` in `load_metadata.py`.

## Integration Points

### Worker Startup (`worker.py`)

```python
# Called at module import time, OUTSIDE any transaction

from aeon.dj_pipeline import acquisition
from aeon.dj_pipeline.utils import streams_maker
from aeon.dj_pipeline.utils.load_metadata import (
    get_experiment_pydantic,
    populate_catalog_from_pydantic,
)

# STEP 1: Populate catalog from all registered Experiment classes
for exp in acquisition.Experiment.DevicesSchema.fetch(as_dict=True):
    experiment_class = get_experiment_pydantic(exp["devices_schema_name"])
    populate_catalog_from_pydantic(experiment_class)

# STEP 2: Create tables (MUST be outside transaction)
streams = streams_maker.main()
```

### EpochConfig.make() (inside transaction)

```python
def make(self, key):
    # 1. Get Experiment class path (e.g., "swc.aeon.exp.foragingABC.experiment:Experiment")
    schema_name = (Experiment.DevicesSchema & key).fetch1("devices_schema_name")

    # 2. Load metadata file and extract rig_config
    metadata = json.loads(metadata_filepath.read_text())
    rig_config = metadata.get("metadata", {}).get("rig", {})

    # 3. Validate and construct Experiment/Rig from rig_config
    experiment_class = get_experiment_pydantic(schema_name)
    experiment = experiment_class.model_validate({"rig": rig_config})
    rig = experiment.rig

    # 4. Store original rig_config in EpochConfig.Meta (as JSON)
    epoch_config = {
        ...
        "metadata": rig_config,  # Stored for Step 3 (data loading)
    }

    # 5. Insert device types, serial numbers, and names (DeviceType, Device, DeviceName)
    insert_device_types(rig, metadata_filepath)

    # 6. Insert device installation records (ExperimentDevice tables)
    ingest_epoch_metadata_from_rig(experiment_name, rig, epoch_config, metadata_filepath)

    # 7. Insert epoch config with metadata
    self.Meta.insert1(epoch_config)

    # Tables already exist (created at worker startup)
    # This method only does DML (inserts), no DDL (table creation)
```

**StreamType handling**: `DeviceType.Stream` has FK to `StreamType`. The `populate_catalog_from_pydantic()` function handles this by inserting `StreamType` entries before `DeviceType.Stream`.

**Generated `streams.py` imports**:
```python
#----                     DO NOT MODIFY                ----
#---- THIS FILE IS AUTO-GENERATED BY `streams_maker.py` ----

import re
import datajoint as dj
import pandas as pd
from uuid import UUID

import aeon
from aeon.dj_pipeline import acquisition, get_schema_name
from swc.aeon.io import api as io_api

schema = dj.Schema(get_schema_name("streams"))
```

## Device vs Stream Distinction

### Pydantic Schema Definition

```python
# Multi-stream device (extends base from swc.aeon.schema.video)
class Camera(SpinnakerCamera):
    trigger: TriggerName = Field(default=TriggerName.TRIGGER0)

    @data_reader
    def video(self, pattern) -> reader.Video:
        return Video(f"{pattern}").reader

    @data_reader
    def position(self, pattern) -> reader.Position:
        return Position(f"{pattern}").reader

# Multi-stream device (extends base from swc.aeon.schema.foraging)
class Feeder(UndergroundFeeder):
    @data_reader
    def beam_break(self, pattern) -> reader.BitmaskEvent:
        return BeamBreak(f"{pattern}").reader

    @data_reader
    def encoder(self, pattern) -> reader.Encoder:
        return Encoder(f"{pattern}").reader
```

### Parsing Logic

The `get_device_info()` function extracts streams from `@data_reader` methods:

```python
# For each device in Rig
device_class = type(device)
stream_types = [name for name, _ in get_data_reader_methods(device_class)]
# Returns: ["video", "position"] (snake_case method names)

# Convert to PascalCase for StreamType catalog
stream_type_names = [to_pascal_case(st) for st in stream_types]
# Returns: ["Video", "Position"]
```

### DataJoint Table Structure

| Component | Table Type | Purpose | Example |
|-----------|-----------|---------|---------|
| **ExperimentDevice** | `dj.Manual` | Track device installation/removal | `Camera` |
| **DeviceDataStream** | `dj.Imported` | Store raw data per chunk | `CameraVideo` |

**ExperimentDevice Table** (`Camera`):
```python
-> Experiment
-> DeviceName
camera_install_time: datetime(6)
---
device_serial_number=null: varchar(12)  # Optional: physical device serial/port
```

**DeviceDataStream Table** (`CameraVideo`):
```python
-> Camera
-> Chunk
---
sample_count: int
timestamps: longblob
hw_counter: longblob
hw_timestamp: longblob
```

## Example: ForagingABC Complete Flow

### 1. Schema Definition (from `aeon_exp_foragingABC/rig.py`)

```python
from swc.aeon.schema import BaseSchema, DataSchema, data_reader
from swc.aeon.schema.video import SpinnakerCamera
from swc.aeon.schema.foraging import UndergroundFeeder
from swc.aeon.io import reader

class Camera(SpinnakerCamera):
    trigger: TriggerName = Field(default=TriggerName.TRIGGER0)
    camera_tracking: Tracking | None = Field(default=None)

    @data_reader
    def video(self, pattern) -> reader.Video:
        return Video(f"{pattern}").reader

    @data_reader
    def position(self, pattern) -> reader.Position:
        if self.camera_tracking is None:
            raise ValueError(f"No tracking defined for {pattern}")
        return Position(f"{pattern}").reader


class Feeder(UndergroundFeeder):
    @data_reader
    def beam_break(self, pattern) -> reader.BitmaskEvent:
        return BeamBreak(f"{pattern}").reader

    @data_reader
    def encoder(self, pattern) -> reader.Encoder:
        return Encoder(f"{pattern}").reader


class Rig(BaseSchema):
    cameras: Dict[CameraName, Camera]           # 13 cameras
    feeders: Dict[FeederName, Feeder]           # 6 feeders
    nest: Dict[NestName, ActivityWeightScale]   # Weight scale
```

### 2. Worker Startup (Step 1 + Step 2)

```python
# worker.py - at module import time

# STEP 1: Populate catalog
experiment_class = get_experiment_pydantic("swc.aeon.exp.foragingABC.experiment:Experiment")
populate_catalog_from_pydantic(experiment_class)

# Result: StreamType, DeviceType, DeviceType.Stream tables populated
# StreamType contains stream_reader path for on-demand column extraction

# STEP 2: Create tables
streams = streams_maker.main()

# Result: Camera, CameraVideo, Feeder, FeederBeamBreak, etc. tables created
```

### 3. EpochConfig.make() (Step 3 - DML only)

```python
# When EpochConfig.populate() runs for a new epoch:

# Parse metadata.json → get device INSTANCES (CameraTop, Feeder1, etc.)
rig = experiment_class.model_validate({"rig": rig_config}).rig

# Insert device types, serial numbers, and names
insert_device_types(rig, metadata_filepath)

# Insert device installations (rows in ExperimentDevice tables)
ingest_epoch_metadata_from_rig(...)

# Store rig_config for data loading
self.Meta.insert1({..., "metadata": rig_config})

# Tables already exist - no DDL needed!
```

### 4. Epoch Config Storage

The original `rig_config` JSON is stored in `EpochConfig.Meta.metadata`:
```json
{
  "cameras": {
    "top": {"serialNumber": "12345", "trigger": "Trigger0", ...},
    "side": {"serialNumber": "12346", "trigger": "Trigger1", ...}
  },
  "feeders": {
    "feeder1": {"portName": "COM3", ...}
  }
}
```

This enables Pydantic reconstruction for data loading (Step 3).

### 5. Data Population (Step 3 - Stream Data)

When `CameraVideo.populate()` is called:
1. `key_source` returns Chunk × Camera combinations
2. For each key, `make()` is called
3. `get_stream_reader_for_epoch()` is called which:
   - Fetches Experiment class path from `Experiment.DevicesSchema`
   - Fetches `rig_config` from `EpochConfig.Meta`
   - Loads Experiment class using `get_experiment_pydantic()`
   - Reconstructs via `Experiment.model_validate({"rig": rig_config})`
   - Accesses `experiment.rig` to get Rig instance
4. Camera device is found in `rig.cameras[device_name]`
5. `camera.video` property returns configured Video reader
6. `io_api.load()` reads data using the reader
7. Data is inserted into the stream table

### 6. Usage

```python
from aeon.dj_pipeline import streams

# Query installed cameras
streams.Camera & {"experiment_name": "foraging-abc"}

# Populate video data for all chunks
streams.CameraVideo.populate()

# Query encoder data
streams.FeederEncoder & {"experiment_name": "foraging-abc"}
```

## Design Decisions

### Why Three Decoupled Steps?

The separation into three steps solves a fundamental constraint: **MySQL forbids DDL inside transactions**.

- `dj.Imported.populate()` wraps `make()` in a transaction
- If `streams_maker.main()` is called inside `make()`, and tables don't exist yet, DDL fails
- By pre-creating tables at worker startup (outside transaction), `make()` only does DML

**Limitation**: If a new experiment is registered AFTER worker startup, the catalog won't have entries for the new device/stream types, and the ExperimentDevice/DeviceDataStream tables won't exist. The worker must be restarted to pick up new experiments.

### Why Extract Columns On-Demand?

Columns are extracted on-demand by importing the reader class from `StreamType.stream_reader`:
- Import the reader class directly (e.g., `"swc.aeon.io.reader.BitmaskEvent"`)
- Instantiate with dummy pattern to get columns (columns are instance attributes)
- No need to store columns in database - they're always available from the class

### Why `stream_hash` as Primary Key?

Different experiments may define the same `stream_type` name (e.g., "Video") with different reader implementations. The `stream_hash` (UUID of stream_type + stream_reader) ensures uniqueness while allowing the catalog to track all variations.

### Why Store `rig_config` as JSON?

1. **Avoids repeated file I/O**: Stream tables' `make()` methods don't need to read metadata files
2. **Enables Pydantic reconstruction**: `model_validate()` can recreate the full Rig from JSON
3. **Preserves nested structure**: Required for proper `@data_reader` pattern resolution
4. **Epoch-specific**: Each epoch stores its own configuration, supporting configuration changes over time

### Why Use `module.path:ClassName` Format?

The `devices_schema_name` uses colon separator (`module.path:ClassName`) instead of dot notation for clarity:
- **Explicit separation**: Clearly distinguishes module path from class name
- **Python convention**: Aligns with entry point syntax (e.g., `console_scripts`)
- **Simpler parsing**: `rsplit(':', 1)` reliably separates module from class
- Example: `"swc.aeon.exp.foragingABC.experiment:Experiment"`

### Why Point to Experiment Class Instead of Rig?

The schema path points to the `Experiment` class (which has `rig: Rig`) rather than the `Rig` class directly because:
1. **Pydantic validation**: `Experiment.model_validate({"rig": rig_config})` validates the full structure
2. **Future extensibility**: Experiment may contain other metadata (e.g., `meta_controller`)
3. **Consistent hierarchy**: Maintains the `Experiment.rig` relationship as defined in the schema

### Why Columns are Class-Level Information?

Stream columns are defined in the reader CLASS, not instance config:

```python
class Video(reader.Harp):
    def __init__(self, pattern):
        super().__init__(pattern, columns=["hw_counter", "hw_timestamp"])  # Fixed!
```

The `pattern` affects WHERE data is read from, not WHAT columns exist. This insight enables extracting columns from the Pydantic class without any epoch metadata.

## Handling Readers with Constructor Kwargs

Some readers require additional constructor arguments beyond `pattern`:

**Affected readers and their required kwargs:**

| Reader | Required Kwargs | Example Usage |
|--------|----------------|---------------|
| `BitmaskEvent` | `value`, `tag` | `BitmaskEvent(pattern, value=0x22, tag="PelletDetected")` |
| `Harp` | `columns` | `Harp(pattern, columns=["retried_delivery"])` |
| `DigitalBitmask` | `mask`, `columns` | `DigitalBitmask(pattern, mask=0x01, columns=["event"])` |
| `Csv` | `columns` (optional) | `Csv(pattern, columns=["channel", "value"])` |

### The Solution: Store `stream_reader_kwargs` in StreamType

Extend the `StreamType` table to store reader constructor kwargs:

```python
class StreamType(dj.Lookup):
    definition = """ # Catalog of unique stream types used across Project Aeon
    stream_hash: uuid  # hash(stream_type, stream_reader) - unique identifier
    ---
    stream_type: varchar(36)
    stream_reader: varchar(256)
    stream_reader_kwargs=null: longblob  # JSON dict of kwargs (excluding pattern)
    stream_description='': varchar(256)
    unique index(stream_type, stream_reader)
    """
```

### Kwargs Extraction Strategy

During Step 1 (`populate_catalog_from_pydantic()`), extract kwargs by executing `@data_reader` methods:

```python
def _extract_kwargs_from_reader(reader) -> dict | None:
    """Extract constructor kwargs from reader instance.

    Uses inspect.signature to determine what kwargs the reader's __init__ accepts,
    then extracts those values from the instance attributes. This dynamic approach
    handles any reader class without hardcoding specific attribute names.

    Args:
        reader: Reader instance from @data_reader method

    Returns:
        Dict of kwargs (excluding 'pattern'), or None if no special kwargs
    """
    reader_class = reader.__class__
    sig = inspect.signature(reader_class.__init__)

    kwargs = {}
    for param_name, param in sig.parameters.items():
        # Skip 'self' and 'pattern' (the required positional args)
        if param_name in ('self', 'pattern'):
            continue

        # Get the value from the instance
        if hasattr(reader, param_name):
            value = getattr(reader, param_name)
            if value is not None:
                # Handle numpy arrays, tuples -> list for JSON serialization
                if hasattr(value, 'tolist'):
                    value = value.tolist()
                elif isinstance(value, tuple):
                    value = list(value)
                kwargs[param_name] = value

    return kwargs if kwargs else None


def get_reader_kwargs_from_device_class(device_class: type, method_name: str) -> dict | None:
    """Extract reader constructor kwargs by executing @data_reader method.

    Creates a minimal device instance using model_construct() (bypasses validation)
    and executes the @data_reader method to get the configured reader.

    Args:
        device_class: Device class type (Pydantic BaseSchema)
        method_name: Name of the @data_reader method (snake_case)

    Returns:
        Dict of kwargs (excluding 'pattern'), or None if extraction fails
    """
    try:
        # Create minimal device instance bypassing validation
        device = device_class.model_construct()

        # Get the @data_reader property (returns reader instance)
        try:
            reader = getattr(device, method_name)
        except (ValueError, AttributeError, TypeError) as e:
            # Method raised error (e.g., camera_tracking is None for position)
            logger.debug(f"Could not execute {method_name} on {device_class.__name__}: {e}")
            return None

        # Extract kwargs from reader instance
        return _extract_kwargs_from_reader(reader)

    except Exception as e:
        logger.debug(f"Failed to extract kwargs for {device_class.__name__}.{method_name}: {e}")
        return None
```

### Usage in Table Creation (Step 2)

```python
def get_device_stream_template(device_type, stream_type, streams_module):
    # Fetch stream info including kwargs
    stream_detail = (
        DeviceType.Stream * StreamType
        & {"device_type": device_type, "stream_type": stream_type}
    ).fetch1()

    stream_reader_path = stream_detail["stream_reader"]
    stream_reader_kwargs = stream_detail.get("stream_reader_kwargs") or {}

    # Import reader class
    module_path, class_name = stream_reader_path.rsplit(".", 1)
    reader_module = importlib.import_module(module_path)
    reader_class = getattr(reader_module, class_name)

    # Instantiate with stored kwargs
    reader_instance = reader_class("_dummy_pattern_", **stream_reader_kwargs)
    columns = reader_instance.columns

    # ... generate table definition using columns
```

### Complete Flow with Kwargs

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Extract and Store Kwargs                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  @data_reader                                                               │
│  def beam_break(self, pattern) -> reader.BitmaskEvent:                      │
│      return BeamBreak(f"{pattern}").reader  # BeamBreak wraps BitmaskEvent  │
│                                                                             │
│              ↓                                                              │
│                                                                             │
│  populate_catalog_from_pydantic() calls:                                    │
│  1. device.model_construct() → minimal device instance                      │
│  2. getattr(device, "beam_break") → reader instance                         │
│  3. _extract_kwargs_from_reader(reader) → {"value": 0x22, "tag": "..."}     │
│                                                                             │
│              ↓                                                              │
│                                                                             │
│  StreamType.insert1({                                                       │
│      "stream_hash": ...,                                                    │
│      "stream_type": "BeamBreak",                                            │
│      "stream_reader": "swc.aeon.io.reader.BitmaskEvent",                    │
│      "stream_reader_kwargs": {"value": 0x22, "tag": "PelletDetected"},      │
│  })                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Use Kwargs for Column Extraction                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  get_device_stream_template():                                              │
│  1. Fetch stream_reader_kwargs from StreamType                              │
│  2. Import reader class from stream_reader path                             │
│  3. reader_class("_dummy_", **stream_reader_kwargs) → success!              │
│  4. Extract columns from reader instance                                    │
│  5. Generate table definition                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Edge Cases

**1. Methods that access `self` attributes:**
```python
@data_reader
def position(self, pattern) -> reader.Position:
    if self.camera_tracking is None:  # Accesses self.camera_tracking
        raise ValueError(f"No tracking defined for {pattern}")
    return Position(f"{pattern}").reader
```
- **Solution**: Catch exceptions and return `None` for kwargs
- These streams will have `stream_reader_kwargs=null`
- Table creation will attempt to instantiate without kwargs (may fail gracefully)

**2. Readers with no special kwargs (e.g., Video, Encoder):**
- `_extract_kwargs_from_reader()` returns `None` or `{}`
- `stream_reader_kwargs` stored as `null`
- Table creation instantiates with just pattern (current behavior)

**3. Kwargs changes between experiments:**
- Different experiments may use same stream_type with different kwargs
- The `stream_hash` does NOT include kwargs (only stream_type + stream_reader)
- Same (stream_type, stream_reader) combination shares one entry regardless of kwargs
- This works because columns are determined by the reader class, not kwargs values
