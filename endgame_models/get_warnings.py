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


def compare_dicts(data_dict: DictType, model_dict: DictType) -> DictMatchType:
    def compare_iterable(
        list1: ListType | TupleType, list2: ListType | TupleType
    ) -> ListMatchType:
        output_list = []
        for item, other_item in zip(list1, list2):
            if isinstance(item, dict):
                assert isinstance(other_item, dict)
                out = compare_dicts(item, other_item)
            elif isinstance(item, (list, tuple)):
                assert isinstance(other_item, type(item))
                out = compare_iterable(item, other_item)
            else:
                out = None
            if out is not None:
                output_list += [out]
        return output_list

    difference_dict = {}
    for k, v in data_dict.items():
        # TODO: Why does below line break stuff
        # if (other_v := model_dict.get(k)) and v != other_v:
        if k in model_dict:
            other_v = model_dict[k]
            if v != other_v:
                if isinstance(v, dict):
                    assert isinstance(other_v, dict)
                    difference_dict[k] = compare_dicts(v, other_v)
                elif isinstance(v, (list, tuple)):
                    assert isinstance(other_v, type(v))
                    difference_dict[k] = compare_iterable(v, other_v)
        elif m := get_close_matches(k, model_dict.keys()):
            difference_dict[k] = DifferenceData(m)

    return difference_dict


def outer_flatten(obj: DictMatchType, prefix: str = "") -> list[str]:
    def flatten(
        obj: DictMatchType | ListMatchType, prefix: str = ""
    ) -> dict[str, DifferenceData]:
        iterator = obj.items() if isinstance(obj, dict) else enumerate(obj)
        ret = {}
        for k, v in iterator:
            if isinstance(v, dict) or isinstance(v, list):
                update = flatten(v, prefix=f"{prefix} -> {k}")
            else:
                update = {f"Close match detected for: \n{prefix} -> {k}": v}
            ret.update(update)
        return ret

    flat = flatten(obj, prefix=prefix)

    return [
        f'{k} - Did you mean {" or ".join(v.matches)}?'
        for k, v in flat.items()
        if isinstance(v, DifferenceData)
    ]


def get_warnings(data_dict: DictType, model_dict: DictType, prefix: str) -> list[str]:
    return outer_flatten(compare_dicts(data_dict, model_dict), prefix=prefix)
