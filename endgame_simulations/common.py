from abc import ABC, abstractclassmethod, abstractmethod
from typing import Generic, Protocol, TypeVar

import h5py
import tqdm
from hdf5_dataclass import FileType

from endgame_simulations.models import (
    BaseInitialParams,
    BaseProgramParams,
    EndgameModel,
    InitialParams,
    _BaseUpdateParams,
)

StateParams = TypeVar("StateParams", bound=BaseInitialParams)


class BaseState(Generic[StateParams], ABC):
    current_time: float

    @abstractclassmethod
    def from_params(
        cls,
        params: StateParams,
        current_time: float = 0.0,
    ):
        ...

    @abstractclassmethod
    def from_hdf5(
        cls,
        params: StateParams,
        current_time: float = 0.0,
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

    def __eq__(self, other: object) -> bool:
        ...


State = TypeVar("State", bound=BaseState, contravariant=True)


class AdvanceState(Protocol, Generic[State]):
    def __call__(self, state: State, debug: bool = False) -> None:
        ...
