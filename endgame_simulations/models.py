__all__ = [
    "EndgameModel",
    "create_update_model",
    "apply_incremental_param_changes",
    "read_only",
    "BaseInitialParams",
    "BaseProgramParams",
]

import warnings
from typing import Generic, Iterable, Optional, TypeVar, get_origin

from pydantic import BaseModel, Field, PrivateAttr, create_model
from pydantic.fields import FieldInfo
from pydantic.generics import GenericModel
from pydantic.main import ModelMetaclass
from typing_extensions import dataclass_transform

from .get_warnings import get_warnings


def read_only(default=...):
    return Field(default=default, allow_mutation=False)


# NOTE: This metaclass exists to allow us to use dataclass transform. Dataclass
# transform lets you make a pydantic class with new field specifiers. The idea is that
# then "read only" can be used to exclude certain parameters from the update model.
@dataclass_transform(
    kw_only_default=True, field_specifiers=(read_only, Field, FieldInfo)
)
class _MetaParams(ModelMetaclass):
    pass


class BaseParams(BaseModel, metaclass=_MetaParams):
    """
    The abstract base class for all parameters. Validate assignment
    is used to say that when mutated, the model will be re-validated.
    """

    class Config:
        validate_assignment = True


class BaseEndgame(BaseModel):
    """
    The Base class for endgame models created.
    """

    def __init__(self, **data) -> None:
        super().__init__(**data)
        new_data = {}
        for k, v in data.items():
            if isinstance(v, BaseModel):
                new_data[k] = v.dict()
            else:
                new_data[k] = v
        output = get_warnings(new_data, self.dict(), prefix=self.__class__.__name__)
        for w in output:
            warnings.warn(w, stacklevel=9999)


class BaseInitialParams(BaseParams):
    """
    The base class for the "initial" section of the endgame model. Custom models must inherit
    from this - used for typing.
    """


class _BaseUpdateParams(BaseParams):
    """
    The base class for the "changes" section of the endgame model. Custom models must inherit
    from this - used for typing. This is typically used via "create_update_model"
    """


def create_update_model(
    initial_model: type[BaseInitialParams],
) -> type[_BaseUpdateParams]:
    """
    Creates a model for the "changes" section of the endgame model, from the model of the "initial"
    section of the endgame model.

    Args:
        initial_model (type[BaseInitialParams]): the model of the "initial" section of the endgame model.

    Returns:
        type[_BaseUpdateParams]: The model for the "changes" section of the endgame model
    """
    new_fields = {}
    for field in initial_model.__fields__.values():
        if field.field_info.allow_mutation:
            if get_origin(field.outer_type_) != list and issubclass(
                field.outer_type_, BaseInitialParams
            ):
                new_model = create_update_model(field.outer_type_)
            else:
                new_model = field.outer_type_
            new_fields[field.name] = (Optional[new_model], None)
    return create_model(
        f"Update{initial_model.__name__}", __base__=_BaseUpdateParams, **new_fields
    )


InitialParams = TypeVar("InitialParams", bound=BaseInitialParams)
UpdateParams = TypeVar("UpdateParams", bound=_BaseUpdateParams)


class ParameterChange(GenericModel, Generic[UpdateParams]):
    """
    The model underpinning each change in the endgame model. Follows
    a set structure.
    """

    year: int
    month: int = 1
    params: UpdateParams


class Parameters(GenericModel, Generic[InitialParams, UpdateParams]):
    """
    The "parameters" section of the endgame model
    """

    initial: InitialParams
    changes: list[ParameterChange[UpdateParams]]


class BaseProgramParams(BaseParams):
    treatment_interval: float
    treatment_name: str = "IVM"


ProgramParams = TypeVar("ProgramParams", bound=BaseProgramParams)


class Program(GenericModel, Generic[ProgramParams]):
    """
    The model representing each items in the "programs" part of the endgame structure.
    """

    first_year: int
    first_month: int = 1
    last_year: int
    last_month: int = 12
    interventions: ProgramParams | list[ProgramParams]


class EndgameModel(
    GenericModel, Generic[InitialParams, UpdateParams, ProgramParams], BaseEndgame
):
    """
    The Base Model for all endgame models. Receives:

    InitialParams: The model representing the "initial" part of the endgame structure.
    UpdateParams: The model representing each item in the "changes" part of the endgame
        structure.
    ProgramParams: The model representing each item in the "interventions" part of the endgame
        structure.
    """

    __top__ = PrivateAttr()
    parameters: Parameters[InitialParams, UpdateParams]
    programs: list[Program[ProgramParams]]


def apply_incremental_param_changes(
    initial: InitialParams,
    changes: ParameterChange[_BaseUpdateParams]
    | Iterable[ParameterChange[_BaseUpdateParams]],
) -> InitialParams:
    """
    Used to apply a change or series of changes to an initial set of parameters. Produces a model with the changes
    applied in the sequence given.

    Args:
        initial (InitialParams): The starting set of parameters.
        changes (ParameterChange[_BaseUpdateParams] | Iterable[ParameterChange[_BaseUpdateParams]]): The change(s) applied
            to this initial set, in the order given.

    Returns:
        InitialParams: The resulting set of updated parameters.
    """
    current_dict = initial.dict()
    if isinstance(changes, ParameterChange):
        current_dict.update(changes.params.dict(exclude_unset=True))
    else:
        for change in changes:
            current_dict.update(change.params.dict(exclude_unset=True))
    return type(initial).parse_obj(current_dict)


def make_endgame_model(
    name: str,
    initial_model: type[InitialParams],
    treatment_model: type[ProgramParams],
) -> type[EndgameModel[InitialParams, _BaseUpdateParams, ProgramParams]]:
    """
    Makes a new endgame model based on an initial model and the treatment model.

    Args:
        name (str): The name of the new model.
        initial_model (type[InitialParams]): The model representing the "initial" part of the endgame structure.
        treatment_model (type[ProgramParams]): The model representing each item in the "interventions" part of the
            endgame structure.

    Returns:
        type[EndgameModel[InitialParams, _BaseUpdateParams, ProgramParams]]: The created endgame model.
    """
    NewModel = EndgameModel[
        initial_model, create_update_model(initial_model), treatment_model
    ]
    NewModel.__name__ = name
    return NewModel
