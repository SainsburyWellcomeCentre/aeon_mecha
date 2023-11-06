import json
import os
import time
from datetime import datetime, timedelta

import datajoint as dj
import requests

from . import get_schema_name, lab

logger = dj.logger
schema = dj.schema(get_schema_name("subject"))


@schema
class Strain(dj.Manual):
    definition = """
    strain_id: int
    ---
    strain_name: varchar(64)
    """


@schema
class GeneticBackground(dj.Manual):
    definition = """
    gen_bg_id: int
    ---
    gen_bg: varchar(64)
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


@schema
class SubjectDetail(dj.Imported):
    definition = """
    -> Subject
    ---
    lab_id='': varchar(128)  # pyrat 'labid'
    responsible_fullname='': varchar(128)
    -> [nullable] GeneticBackground
    -> [nullable] Strain
    cage_number='': varchar(32)
    available=1: bool  # is this animal available on pyrat
    """

    def make(self, key):
        eartag_or_id = key["subject"]
        # cage id, sex, line/strain, genetic background, dob, lab id
        params = {
            "k": _pyrat_animal_attributes,
            "s": "eartag_or_id:asc",
            "o": 0,
            "l": 10,
            "eartag": eartag_or_id,
        }
        animal_resp = get_pyrat_data(endpoint="animals", params=params)
        if len(animal_resp) == 0:
            if self & key:
                self.update1(
                    {
                        **key,
                        "available": False,
                    }
                )
            else:
                self.insert1(
                    {
                        **key,
                        "available": False,
                    }
                )
            return
        elif len(animal_resp) > 1:
            raise ValueError(f"Found {len(animal_resp)} with eartag {eartag_or_id}, expect one")
        else:
            animal_resp = animal_resp[0]

        # Insert new subject
        Subject.update1(
            {
                **key,
                "sex": {"f": "F", "m": "M", "?": "U"}[animal_resp["sex"]],
                "subject_birth_date": animal_resp["dateborn"],
            }
        )
        Strain.insert1(
            {"strain_id": animal_resp["strain_id"], "strain_name": animal_resp["strain_id"]},
            skip_duplicates=True,
        )
        entry = {
            **key,
            "responsible_fullname": animal_resp["responsible_fullname"],
            "strain_id": animal_resp["strain_id"],
            "cage_number": animal_resp["cagenumber"],
            "lab_id": animal_resp["labid"],
        }
        if animal_resp["gen_bg_id"] is not None:
            GeneticBackground.insert1(
                {"gen_bg_id": animal_resp["gen_bg_id"], "gen_bg": animal_resp["gen_bg"]},
                skip_duplicates=True,
            )
            entry["gen_bg_id"] = animal_resp["gen_bg_id"]

        self.insert1(entry)


@schema
class SubjectWeight(dj.Imported):
    definition = """
    -> Subject
    weight_id: int
    ---
    weight: float
    weight_time: datetime
    actor_name: varchar(200)
    """


@schema
class SubjectProcedure(dj.Imported):
    definition = """
    -> Subject
    assign_id: int
    ---
    procedure_id: int
    procedure_name: varchar(200)
    procedure_date: date
    license_id=null: int
    license_number=null: varchar(200)
    classification_id=null: int
    classification_name=null: varchar(200)
    actor_fullname: varchar(200)
    comment=null: varchar(1000)
    """


@schema
class SubjectComment(dj.Imported):
    definition = """
    -> Subject
    comment_id: int
    ---
    created: datetime
    creator_id: int
    creator_username: varchar(200)
    creator_fullname: varchar(200)
    origin: varchar(200)
    content=null: varchar(1000)
    attributes: varchar(1000)
    """


@schema
class SubjectReferenceWeight(dj.Manual):
    definition = """
    -> SubjectDetail
    ---
    reference_weight=null: float  # animal's reference weight for use in experiment
    last_updated_time: datetime   # last time (in UTC) when the reference weight was updated
    """

    @classmethod
    def get_reference_weight(cls, subject_name):
        subj_key = {"subject": subject_name}

        food_restrict_query = SubjectProcedure & subj_key & "procedure_name = 'A99 - food restriction'"
        if food_restrict_query:
            ref_date = food_restrict_query.fetch("procedure_date", order_by="procedure_date DESC", limit=1)[
                0
            ]
        else:
            ref_date = datetime.now().date()

        weight_query = SubjectWeight & subj_key & f"weight_time < '{ref_date}'"
        ref_weight = (
            weight_query.fetch("weight", order_by="weight_time DESC", limit=1)[0] if weight_query else None
        )

        entry = {
            "subject": subject_name,
            "reference_weight": ref_weight,
            "last_updated_time": datetime.utcnow(),
        }
        cls.update1(entry) if cls & {"subject": subject_name} else cls.insert1(entry)

        return ref_weight


# ------------------- PYRAT SYNCHRONIZATION --------------------


@schema
class PyratIngestionTask(dj.Manual):
    """Task to sync new animals from PyRAT."""

    definition = """
    pyrat_task_scheduled_time: datetime  # (UTC) scheduled time for task execution
    """


@schema
class PyratIngestion(dj.Imported):
    """Ingestion of new animals from PyRAT."""

    definition = """
    -> PyratIngestionTask
    ---
    execution_time: datetime  # (UTC) time of task execution
    execution_duration: float  # (s) duration of task execution
    new_pyrat_entry_count: int  # number of new PyRAT subject ingested in this round of ingestion
    """

    key_source = (
        PyratIngestionTask.proj(
            seconds_since_scheduled="TIMESTAMPDIFF(SECOND, pyrat_task_scheduled_time, UTC_TIMESTAMP())"
        )
        & "seconds_since_scheduled >= 0"
    )

    auto_schedule = True
    schedule_interval = 12  # schedule interval in number of hours

    def _auto_schedule(self):
        utc_now = datetime.utcnow()

        next_task_schedule_time = utc_now + timedelta(hours=self.schedule_interval)
        if (
            PyratIngestionTask
            & f"pyrat_task_scheduled_time BETWEEN '{utc_now}' AND '{next_task_schedule_time}'"
        ):
            return

        PyratIngestionTask.insert1({"pyrat_task_scheduled_time": next_task_schedule_time})

    def make(self, key):
        execution_time = datetime.utcnow()
        """Automatically import or update entries in the Subject table."""
        new_eartags = []
        for responsible_id in lab.User.fetch("responsible_id"):
            # 1 - retrieve all animals from this user
            animal_resp = get_pyrat_data(endpoint="animals", params={"responsible_id": responsible_id})
            for animal_entry in animal_resp:
                # 2 - find animal with comment - Project Aeon
                eartag_or_id = animal_entry["eartag_or_id"]
                comment_resp = get_pyrat_data(endpoint=f"animals/{eartag_or_id}/comments")
                for comment in comment_resp:
                    if comment["attributes"]:
                        first_attr = comment["attributes"][0]
                        if (
                            first_attr["label"].lower() == "project"
                            and first_attr["content"].lower() == "aeon"
                        ):
                            new_eartags.append(eartag_or_id)

        new_entry_count = 0
        for eartag_or_id in new_eartags:
            if Subject & {"subject": eartag_or_id}:
                continue
            Subject.insert1(
                {
                    "subject": eartag_or_id,
                    "sex": "U",
                    "subject_birth_date": "1900-01-01",
                }
            )
            new_entry_count += 1

        logger.info(f"Inserting {new_entry_count} new subject(s) from Pyrat")
        completion_time = datetime.utcnow()
        self.insert1(
            {
                **key,
                "execution_time": execution_time,
                "execution_duration": (completion_time - execution_time).total_seconds(),
                "new_pyrat_entry_count": new_entry_count,
            }
        )

        # auto schedule next task
        if self.auto_schedule:
            self._auto_schedule()


@schema
class PyratCommentWeightProcedure(dj.Imported):
    """Ingestion of new animals from PyRAT."""

    definition = """
    -> PyratIngestion
    -> SubjectDetail
    ---
    execution_time: datetime  # (UTC) time of task execution
    execution_duration: float  # (s) duration of task execution
    """

    key_source = (PyratIngestion * SubjectDetail) & "available = 1"

    def make(self, key):
        execution_time = datetime.utcnow()
        logger.info("Extracting weights/comments/procedures")

        eartag_or_id = key["subject"]
        try:
            comment_resp = get_pyrat_data(endpoint=f"animals/{eartag_or_id}/comments")
        except requests.exceptions.HTTPError as e:
            if e.args[0].endswith("response code: 404"):
                SubjectDetail.update1(
                    {
                        **key,
                        "available": False,
                    }
                )
            else:
                raise e
        else:
            for cmt in comment_resp:
                cmt["subject"] = eartag_or_id
                cmt["attributes"] = json.dumps(cmt["attributes"], default=str)
            SubjectComment.insert(comment_resp, skip_duplicates=True, allow_direct_insert=True)

            weight_resp = get_pyrat_data(endpoint=f"animals/{eartag_or_id}/weights")
            SubjectWeight.insert(
                [{**v, "subject": eartag_or_id} for v in weight_resp],
                skip_duplicates=True,
                allow_direct_insert=True,
            )

            procedure_resp = get_pyrat_data(endpoint=f"animals/{eartag_or_id}/procedures")
            SubjectProcedure.insert(
                [{**v, "subject": eartag_or_id} for v in procedure_resp],
                skip_duplicates=True,
                allow_direct_insert=True,
            )

            # compute/update reference weight
            SubjectReferenceWeight.get_reference_weight(eartag_or_id)
        finally:
            completion_time = datetime.utcnow()
            self.insert1(
                {
                    **key,
                    "execution_time": execution_time,
                    "execution_duration": (completion_time - execution_time).total_seconds(),
                }
            )


@schema
class CreatePyratIngestionTask(dj.Computed):
    definition = """
    -> lab.User
    """

    def make(self, key):
        """Create one new PyratIngestionTask for every newly added users."""
        PyratIngestionTask.insert1({"pyrat_task_scheduled_time": datetime.utcnow()})
        time.sleep(1)
        self.insert1(key)


_pyrat_animal_attributes = [
    "animalid",
    "pupid",
    "eartag_or_id",
    "prefix",
    "state",
    "labid",
    "cohort_id",
    "rfid",
    "origin_name",
    "sex",
    "species_name",
    "species_weight_unit",
    "sacrifice_reason_name",
    "sacrifice_comment",
    "sacrifice_actor_username",
    "sacrifice_actor_fullname",
    "datesacrificed",
    "cagenumber",
    "cagetype",
    "cagelabel",
    "cage_owner_username",
    "cage_owner_fullname",
    "rack_description",
    "room_name",
    "area_name",
    "building_name",
    "responsible_id",
    "responsible_fullname",
    "owner_userid",
    "owner_username",
    "owner_fullname",
    "age_days",
    "age_weeks",
    "dateborn",
    "comments",
    "date_last_comment",
    "generation",
    "gen_bg_id",
    "gen_bg",
    "strain_id",
    "strain_name",
    "strain_name_id",
    "strain_name_with_id",
    "mutations",
    "genetically_modified",
    "parents",
    "licence_title",
    "licence_id",
    "licence_number",
    "classification_id",
    "classification",
    "pregnant_days",
    "plug_date",
    "wean_date",
    "projects",
    "requests",
    "weight",
    "animal_color",
    "animal_user_color",
    "import_order_request_id",
]

def get_pyrat_data(endpoint: str, params: dict = None, **kwargs):
    base_url = "https://swc.pyrat.cloud/api/v3/"
    pyrat_system_token = os.getenv("PYRAT_SYSTEM_TOKEN")
    pyrat_user_token = os.getenv("PYRAT_USER_TOKEN")

    if pyrat_system_token is None or pyrat_user_token is None:
        raise ValueError(
            "The PYRAT tokens must be defined as an environment variable named 'PYRAT_SYSTEM_TOKEN' and 'PYRAT_USER_TOKEN'"
        )

    session = requests.Session()
    session.auth = (pyrat_system_token, pyrat_user_token)

    if params is not None:
        params_str_list = []
        for k, v in params.items():
            if isinstance(v, (list, tuple)):
                for i in v:
                    params_str_list.append(f"{k}={i}")
            else:
                params_str_list.append(f"{k}={v}")
        params_str = "?" + "&".join(params_str_list)
    else:
        params_str = ""

    response = session.get(base_url + endpoint + params_str, **kwargs)

    if response.status_code != 200:
        raise requests.exceptions.HTTPError(f'PyRat API errored out with response code: {response.status_code}')

    return response.json()
