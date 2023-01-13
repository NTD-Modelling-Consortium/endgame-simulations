from enum import Enum

from endgame_models import (
    BaseInitialParams,
    BaseProgramParams,
    EndgameModel,
    convert_pydantic,
    reduce_parameters,
)

test = {
    "parameters": {
        "initial": {"w_rate": 0.1, "delta_time": 3},
        "changes": [
            {"year": 2020, "month": 1, "params": {"delta_time": 1}},
            {"year": 2022, "month": 1, "params": {"delta_time": 2, "test": 1}},
        ],
    },
    "programs": [
        {
            "first_year": 2020,
            "first_month": 1,
            "last_year": 2022,
            "last_month": 12,
            "interventions": {
                "min_age": 4,
                "max_age": 5,
                "coverage": 0.7,
                "treatment_interval": 0.1,
                "drug_efficacy": 0.85,
            },
        },
        {
            "first_year": 2026,
            "last_year": 2029,
            "drug_efficacy": 0.50,
            "interventions": [
                {
                    "min_age": 2,
                    "max_age": 5,
                    "coverage": 0.8,
                    "treatment_interval": 0.05,
                    "type": "vaccine",
                },
                {
                    "min_age": 2,
                    "max_age": 5,
                    "coverage": 0.8,
                    "treatment_interval": 0.05,
                    "type": "MDA",
                },
            ],
        },
    ],
}


class TestParams(BaseInitialParams):
    w_rate: int
    delta_time: int


class VaccineType(Enum):
    MDA = "MDA"
    vaccine = "vaccine"


class TestProgramParams(BaseProgramParams):
    min_age: int
    max_age: int
    coverage: float
    type: VaccineType = VaccineType.MDA


NewModel = EndgameModel[TestParams, convert_pydantic(TestParams), TestProgramParams]
output = NewModel.parse_obj(test)
print(output.parameters)


reduced = reduce_parameters(output.parameters.initial, output.parameters.changes)
print(reduced)
