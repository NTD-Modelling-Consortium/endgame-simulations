__all__ = ["get_warnings"]

from dataclasses import dataclass
from difflib import get_close_matches

Flat = str | float | int | bool

JSONType = Flat | list["JSONType"] | dict[str, "JSONType"] | tuple["JSONType", ...]
DictType = dict[str, JSONType]
ListType = list[JSONType]
TupleType = tuple[JSONType, ...]


@dataclass
class DifferenceData:
    matches: list[str]


_MatchType = DifferenceData | list["_MatchType"] | dict[str, "_MatchType"]
ListMatchType = list["_MatchType"]
DictMatchType = dict[str, "_MatchType"]


def _pydantic_similarities(
    raw_dict: DictType, validated_dict: DictType
) -> DictMatchType:
    """
    Compare the processed result of a pydantic model, with the original input data.
    Returns information about the fields that are similar to the model, but excluded
    for having a a different key.

    Args:
        raw_dict (DictType): The original input data dict to the pydantic model.
        validated_dict (DictType): The processed data dict.

    Returns:
        DictMatchType: Tiered dict displaying the location of similar keys.
    """

    def compare_iterable(
        list1: ListType | TupleType, list2: ListType | TupleType
    ) -> ListMatchType:
        output_list = []
        for item, other_item in zip(list1, list2):
            if isinstance(item, dict):
                assert isinstance(other_item, dict)
                out = _pydantic_similarities(item, other_item)
            elif isinstance(item, (list, tuple)):
                assert isinstance(other_item, type(item))
                out = compare_iterable(item, other_item)
            else:
                out = None
            if out is not None:
                output_list += [out]
        return output_list

    difference_dict = {}
    for k, v in raw_dict.items():
        # TODO: Why does below line break stuff
        # if (other_v := validated_dict.get(k)) and v != other_v:
        if k in validated_dict:
            other_v = validated_dict[k]
            if v != other_v:
                if isinstance(v, dict):
                    assert isinstance(other_v, dict)
                    difference_dict[k] = _pydantic_similarities(v, other_v)
                elif isinstance(v, (list, tuple)):
                    assert isinstance(other_v, type(v))
                    difference_dict[k] = compare_iterable(v, other_v)
        elif m := get_close_matches(k, validated_dict.keys()):
            difference_dict[k] = DifferenceData(m)

    return difference_dict


def _flatten(obj: DictMatchType, prefix: str = "") -> list[str]:
    """
    Flatten a series of nested matches from _pydantic_similarities

    Args:
        obj (DictMatchType): Output of _pydantic_similarities
        prefix (str, optional): Prefix of address. Defaults to "".

    Returns:
        list[str]: List of warnings a user can to raise.
    """

    def inner_flatten(
        obj: DictMatchType | ListMatchType, prefix: str = ""
    ) -> dict[str, DifferenceData]:
        iterator = obj.items() if isinstance(obj, dict) else enumerate(obj)
        ret = {}
        for k, v in iterator:
            if isinstance(v, dict) or isinstance(v, list):
                update = inner_flatten(v, prefix=f"{prefix} -> {k}")
            else:
                update = {f"Close match detected for: \n{prefix} -> {k}": v}
            ret.update(update)
        return ret

    flat = inner_flatten(obj, prefix=prefix)

    return [
        f'{k} - Did you mean {" or ".join(v.matches)}?'
        for k, v in flat.items()
        if isinstance(v, DifferenceData)
    ]


def get_warnings(
    raw_dict: DictType, validated_dict: DictType, prefix: str
) -> list[str]:
    """
    Compare the processed result of a pydantic model, with the original input data.
    Returns a list of warnings that a user can raise, based on fields that are similar.

    Args:
        raw_dict (DictType): The original input data dict to the pydantic model.
        validated_dict (DictType): The processed data dict.
        prefix (str): Prefix of address. Typically the class name.

    Returns: List of warnings that a user can raise, based on fields that are similar.
    """
    return _flatten(_pydantic_similarities(raw_dict, validated_dict), prefix=prefix)
