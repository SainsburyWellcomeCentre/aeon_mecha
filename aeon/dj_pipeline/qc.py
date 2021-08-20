import datajoint as dj

from aeon.preprocess import api as aeon_api
from aeon.util import plotting as aeon_plotting

from . import lab, acquisition, tracking
from . import get_schema_name


schema = dj.schema(get_schema_name('qc'))

# -------------- Quality Control ---------------------


@schema
class QCCode(dj.Lookup):
    definition = """
    qc_code: int
    ---
    qc_code_description: varchar(255)
    """


@schema
class QCRoutine(dj.Lookup):
    definition = """
    qc_routine: varchar(24)  # name of this quality control evaluation - e.g. drop_frame
    ---
    qc_routine_order: int    # the order in which this qc routine is executed
    qc_routine_description: varchar(255)  # description of this QC routine
    qc_module: varchar(64)     # the module where the qc_function can be imported from - e.g. aeon.analysis.quality_control
    qc_function: varchar(64)   # the function used to evaluate this QC - e.g. check_drop_frame
    """


# -------------- Data stream level Quality Control ---------------------


@schema
class CameraQC(dj.Computed):
    definition = """ # Quality controls performed on a particular camera for a particular acquisition chunk
    -> acquisition.Chunk
    -> acquisition.ExperimentCamera
    """

    class Routine(dj.Part):
        definition = """  # Quality control routine performed on this session
        -> master
        -> QCRoutine
        ---
        -> QCCode
        qc_comment: varchar(255)  
        """

    class BadPeriod(dj.Part):
        definition = """
        -> master.Routine
        bad_period_start: datetime(6)
        ---
        bad_period_end: datetime(6)
        """


@schema
class FoodPatchQC(dj.Computed):
    definition = """ # Quality controls performed on a particular camera for a particular acquisition chunk
    -> acquisition.Chunk
    -> acquisition.ExperimentFoodPatch
    """

    class Routine(dj.Part):
        definition = """  # Quality control routine performed on this session
        -> master
        -> QCRoutine
        ---
        -> QCCode
        qc_comment: varchar(255)  
        """

    class BadPeriod(dj.Part):
        definition = """
        -> master.Routine
        bad_period_start: datetime(6)
        ---
        bad_period_end: datetime(6)
        """


# -------------- Session level Quality Control ---------------------


@schema
class SessionQC(dj.Computed):
    definition = """  # Quality controls performed on this session
    -> acquisition.Session
    """

    class Routine(dj.Part):
        definition = """  # Quality control routine performed on this session
        -> master
        -> QCRoutine
        ---
        -> QCCode
        qc_comment: varchar(255)  
        """

    class BadPeriod(dj.Part):
        definition = """
        -> master.Routine
        bad_period_start: datetime(6)
        ---
        bad_period_end: datetime(6)
        """

    def make(self, key):
        # depending on which qc_routine
        # fetch relevant data from upstream
        # import the qc_function from the qc_module
        # call the qc_function - expecting a qc_code back, and a list of bad-periods
        # store qc results
        pass


@schema
class BadSession(dj.Computed):
    definition = """  # Session labelled as BadSession and excluded from further analysis
    -> acquisition.Session
    ---
    comment='': varchar(255)  # any comments for why this is a bad session - e.g. manually flagged
    """


@schema
class GoodSession(dj.Computed):
    definition = """  # Quality controls performed on this session
    -> SessionQC
    ---
    qc_routines: varchar(255)  # concatenated list of all the QC routines used for this good/bad conclusion
    """

    class BadPeriod(dj.Part):
        definition = """
        -> master
        bad_period_start: datetime(6)
        ---
        bad_period_end: datetime(6)
        """

    def make(self, key):
        # aggregate all SessionQC results for this session
        # determine Good or Bad Session
        # insert BadPeriod (if none inserted, no bad period)
        pass
