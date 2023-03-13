from abc import ABC, abstractmethod
from typing import Generic, Optional, Protocol, TypeVar

import h5py
from hdf5_dataclass import FileType

from endgame_simulations.models import BaseInitialParams

StateParams = TypeVar("StateParams", bound=BaseInitialParams)


class BaseState(Generic[StateParams], ABC):
    current_time: float
    _previous_delta_time: Optional[float]

    @classmethod
    @abstractmethod
    def from_params(
        cls,
        params: StateParams,
        current_time: float = 0.0,
    ):
        ...

    @classmethod
    @abstractmethod
    def from_hdf5(
        cls,
        input: FileType | h5py.File | h5py.Group,
    ):
        ...

    @abstractmethod
    def to_hdf5(self, output: FileType | h5py.File | h5py.Group):
        """Serialise an object to `output`.
        Use it either to create a new HDF5 file or add to an existing HDF5 File/Group.
        Args:
            output (FileType | h5py.File | h5py.Group): output file/HDF5 group
        """
        ...

    @abstractmethod
    def get_params(self) -> StateParams:
        ...

    @abstractmethod
    def reset_params(self, params: StateParams):
        """Reset the parameters

        Args:
            params (Params): New set of parameters
        """
        ...

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        ...


State = TypeVar("State", bound=BaseState, contravariant=True)


class AdvanceState(Protocol, Generic[State]):
    def __call__(self, state: State, debug: bool = False) -> None:
        ...
