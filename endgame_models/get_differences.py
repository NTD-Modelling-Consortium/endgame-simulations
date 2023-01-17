from dataclasses import dataclass
from typing import Optional, TypeVar

from typing_extensions import Type

from .models import BaseInitialParams

Flat = str | float | int | bool

JSONType = Flat | list["JSONType"] | dict[str, "JSONType"] | tuple["JSONType", ...]
DictType = dict[str, JSONType]
ListType = list[JSONType]
TupleType = tuple[JSONType, ...]


@dataclass
class ReadOnlyDiff:
    original: JSONType
    new: JSONType


OutputReadOnly = dict[str, "ReadOnlyDiff | OutputReadOnly"]


def _output_read_only_diff(
    initial_dict: DictType, update_dict: DictType, model: Type[BaseInitialParams]
) -> Optional[OutputReadOnly]:
    changed_read_only: OutputReadOnly = {}
    for k, new_v in update_dict.items():
        if (old_v := initial_dict.get(k)) is not None:
            assert k in model.__fields__
            # Update and initial both have parameter
            field = model.__fields__[k]
            if not field.field_info.allow_mutation:
                if issubclass(field.outer_type_, BaseInitialParams):
                    assert isinstance(old_v, dict) and isinstance(new_v, dict)
                    out = _output_read_only_diff(old_v, new_v, field.outer_type_)
                    if out is not None:
                        changed_read_only[k] = out
                elif old_v != new_v:
                    changed_read_only[k] = ReadOnlyDiff(old_v, new_v)

    if changed_read_only == {}:
        return None
    else:
        return changed_read_only


def _flatten_output_read_only(
    output_read_only: Optional[OutputReadOnly], prefix: str = ""
) -> list[str]:
    if output_read_only is None:
        return []
    else:

        def inner_flatten(
            output_read_only_inner: OutputReadOnly, prefix: str = ""
        ) -> dict[str, ReadOnlyDiff]:
            ret = {}
            for k, v in output_read_only_inner.items():
                if isinstance(v, dict):
                    update = inner_flatten(v, prefix=f"{prefix} -> {k}")
                else:
                    update = {f"Read only value was changed: \n{prefix} -> {k}": v}
                ret.update(update)
            return ret

        flat = inner_flatten(output_read_only, prefix=prefix)

        return [
            f"{k}   Old value: {v.original} New value: {v.new}"
            for k, v in flat.items()
            if isinstance(v, ReadOnlyDiff)
        ]


ParamType = TypeVar("ParamType", bound=BaseInitialParams)


def get_read_only_differences(
    old_params: ParamType, new_params: ParamType
) -> list[str]:
    """
    Gets the read only properties that have been changed between two sets of parameters.


    Args:
        old_params (ParamType): The old set of parameters
        new_params (ParamType): The new set of parameters

    Returns:
        list[str]: The read only differences between the two.
    """
    model = type(old_params)
    assert type(new_params) == model
    return _flatten_output_read_only(
        _output_read_only_diff(old_params.dict(), new_params.dict(), model),
        model.__name__,
    )
