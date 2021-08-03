import datajoint as dj

from . import lab
from . import get_schema_name


schema = dj.schema(get_schema_name('subject'))


@schema
class Strain(dj.Lookup):
    definition = """
    # Strain of animal, e.g. C57Bl/6
    strain              : varchar(32)	# abbreviated strain name
    ---
    strain_standard_name  : varchar(32)   # formal name of a strain
    strain_desc=''      : varchar(255)	# description of this strain
    """


@schema
class Allele(dj.Lookup):
    definition = """
    allele                      : varchar(32)    # abbreviated allele name
    ---
    allele_standard_name=''     : varchar(255)	  # standard name of an allele
    """

    class Source(dj.Part):
        definition = """
        -> master
        ---
        -> lab.Source
        source_identifier=''        : varchar(255)    # id inside the line provider
        source_url=''               : varchar(255)    # link to the line information
        expression_data_url=''      : varchar(255)    # link to the expression pattern from Allen institute brain atlas
        """


@schema
class Line(dj.Lookup):
    definition = """
    line                    : varchar(32)	# abbreviated name for the line
    ---
    line_description=''     : varchar(2000)
    target_phenotype=''     : varchar(255)
    is_active               : boolean		# whether the line is in active breeding
    """

    class Allele(dj.Part):
        definition = """
        -> master
        -> Allele
        """


@schema
class Subject(dj.Manual):
    definition = """
    # Animal Subject
    subject                 : varchar(32)
    ---
    sex                     : enum('M', 'F', 'U')
    subject_birth_date      : date
    subject_description=''  : varchar(1024)
    """

    class Protocol(dj.Part):
        definition = """
        -> master
        -> lab.Protocol
        """

    class User(dj.Part):
        definition = """
        -> master
        -> lab.User
        """

    class Line(dj.Part):
        definition = """
        -> master
        ---
        -> Line
        """

    class Strain(dj.Part):
        definition = """
        -> master
        ---
        -> Strain
        """

    class Source(dj.Part):
        definition = """
        -> master
        ---
        -> lab.Source
        """

    class Lab(dj.Part):
        definition = """
        -> master
        -> lab.Lab
        ---
        subject_alias=''    : varchar(32)  # alias of the subject in this lab, if different from the id
        """


@schema
class SubjectDeath(dj.Manual):
    definition = """
    -> Subject
    ---
    death_date      : date       # death date
    """


@schema
class SubjectCullMethod(dj.Manual):
    definition = """
    -> Subject
    ---
    cull_method:    varchar(255)
    """


@schema
class Zygosity(dj.Manual):
    definition = """
    -> Subject
    -> Allele
    ---
    zygosity        : enum("Present", "Absent", "Homozygous", "Heterozygous")  # zygosity
    """