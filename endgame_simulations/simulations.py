import warnings
from abc import ABC, abstractproperty
from typing import ClassVar, Generic, Iterator, TypeVar, cast, overload

import h5py
import tqdm
from hdf5_dataclass import FileType
import numpy as np

from endgame_simulations.models import BaseInitialParams

from .common import AdvanceState, BaseState, State

ParamsModel = TypeVar("ParamsModel", bound=BaseInitialParams)


class GenericSimulation(Generic[ParamsModel, State], ABC):
    """
    The Base class for all simulation classes. These represent basic simulations, that are
    simply advanced in time.
    """

    state_class: ClassVar[type[BaseState]]
    advance_state: ClassVar[AdvanceState]
    state: State
    verbose: bool
    debug: bool

    def __init_subclass__(
        cls, *, state_class: type[State], advance_state: AdvanceState
    ) -> None:
        """
        Used when subclassing generic simulation. Note that two keyword arguments are required.

        Args:
            state_class (type[State]): The state class itself is required to instantiate the
                state object on parameter reset.
            advance_state (AdvanceState): The function used for advancing the state.
        """
        # NOTE: The below is a way the state class could be obtained from the type hint, to save the user
        # adding the state class twice.
        # cls.state_class = cls.__orig_bases__[0].__args__[1] #type:ignore
        cls.state_class = state_class
        cls.advance_state = advance_state

    @overload
    def __init__(
        self,
        *,
        start_time: float,
        params: ParamsModel,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Create a new simulation, given the parameters.

        Args:
            start_time (float): Start time of the simulation
            params (ParamsModel): A set of fixed parameters for controlling the model.
            verbose (bool, optional): Verbose?. Defaults to False.
            debug (bool, optional): Debug?. Defaults to False.
        """
        ...

    @overload
    def __init__(
        self,
        *,
        input: FileType | h5py.File | h5py.Group,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Restore the simulation from a previously saved file.

        Args:
            input (FileType | h5py.File | h5py.Group): input file/stream/group
            verbose (bool, optional): Verbose?. Defaults to False.
            debug (bool, optional): Debug?. Defaults to False.
        """
        ...

    def __init__(
        self,
        *,
        start_time: float | None = None,
        params: ParamsModel | None = None,
        input: FileType | h5py.File | h5py.Group | None = None,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        assert (params is not None) != (
            input is not None
        ), "You must provide either `params` or `input`"
        if params:
            state = self.state_class.from_params(params, start_time or 0.0)

        else:
            # input
            assert input is not None
            state = self.state_class.from_hdf5(input)
        self.state = cast(State, state)
        self.verbose = verbose
        self.debug = debug

    @abstractproperty
    def _delta_time(self) -> float:
        ...

    def get_current_params(self) -> ParamsModel:
        return self.state.get_params()

    def reset_current_params(self, params: ParamsModel):
        """Reset the parameters

        Args:
            params (Params): New set of parameters
        """
        self.state.reset_params(params)

    def save(self, output: FileType | h5py.File | h5py.Group) -> None:
        """Save the simulation to a file/stream.

        The output file will be in a HDF5 format. The simulation can then be
        restored with `Simulation.restore` class method.

        Args:
            output (FileType | h5py.File | h5py.Group): output file/stream/group
        """
        self.state.to_hdf5(output)

    @classmethod
    def restore(cls, input: FileType | h5py.File | h5py.Group):
        """Restore the simulation from a file/stream

        Args:
            input (FileType | h5py.File | h5py.Group): HDF5 file/stream/group

        Returns:
            Simulation: restored simulation
        """
        return cls(input=input)

    @overload
    def iter_run(
        self, *, end_time: float, sampling_interval: float, inclusive: bool = False
    ) -> Iterator[State]:
        """Run the simulation until `end_time`. Generates stats every `sampling_interval`,
        until `end_time`.

        This is a generator, so you must it as one.

        Examples:
            >>> simulation = Simulation(start_time=0, params=Params(), n_people=400)
            >>> [sample.mf_prevalence_in_population() for sample in simulation.iter_run(end_time=3, sampling_interval=1.0)]
            [0.99, 0.6, 0.2]

        Args:
            end_time (float): end time
            sampling_interval (float): State sampling interval (years)
            inclusive (bool, optional): If samples include the final end time. Defaults to False.
                Note: technically this just adds delta time to end time

        Yields:
            Iterator[State]: Iterator of the simulation's state.
        """
        ...

    @overload
    def iter_run(
        self, *, end_time: float, sampling_years: list[float], inclusive: bool = False
    ) -> Iterator[State]:
        """Run the simulation until `end_time`. Generates stats for every year in `sampling_years`.

        This is a generator, so you must it as one.

        Examples:
            >>> simulation = Simulation(start_time=0, params=Params(), n_people=400)
            >>> for state in simulation.iter_run(end_time=10, sampling_years=[0.1, 1, 5])
            ...    print(state.mf_prevalence_in_population())
            0.99
            0.6
            0.2

        Args:
            end_time (float): end time
            sampling_years (list[float]): list of years to sample State
            inclusive (bool, optional): If samples include the final end time. Defaults to False.
                Note: technically this just adds delta time to end time

        Yields:
            Iterator[State]: Iterator of the simulation's state.
        """
        ...

    @overload
    def iter_run(
        self,
        *,
        end_time: float,
        sampling_interval: float | None = None,
        sampling_years: list[float] | None = None,
        inclusive: bool = False,
    ) -> Iterator[State]:
        ...

    def iter_run(
        self,
        *,
        end_time: float,
        sampling_interval: float | None = None,
        sampling_years: list[float] | None = None,
        inclusive: bool = False,
    ) -> Iterator[State]:
        if inclusive:
            real_end_time = end_time + self._delta_time
        else:
            real_end_time = end_time
        if real_end_time < self.state.current_time:
            raise ValueError(
                f"End time {real_end_time} before start {self.state.current_time}"
            )

        if sampling_interval and sampling_years:
            raise ValueError(
                "You must provide sampling_interval, sampling_years or neither"
            )

        if sampling_years:
            sampling_years = sorted(sampling_years)

        if sampling_interval is not None:
            sampling_years = np.arange(self.state.current_time, real_end_time, sampling_interval)

        sampling_years_idx = 0

        with tqdm.tqdm(
            total=real_end_time - self.state.current_time + self._delta_time,
            disable=not self.verbose,
        ) as progress_bar:
            while self.state.current_time + self._delta_time <= real_end_time:
                is_on_sampling_year = (
                    sampling_years is not None
                    and sampling_years_idx < len(sampling_years)
                    and self.state.current_time - sampling_years[sampling_years_idx]
                    >= 0
                )
                if is_on_sampling_year:
                    yield self.state
                    sampling_years_idx += 1

                self.state.current_time += self._delta_time
                # crude self-correction at the end of each year to account for floating-point precision issues
                if round(self.state.current_time, 9) % 1 == 0:
                    self.state.current_time = round(self.state.current_time, 9)

                progress_bar.update(self._delta_time)
                type(self).advance_state(self.state, self.debug)
                self.state._previous_delta_time = self._delta_time

    def run(self, *, end_time: float) -> None:
        """Run simulation from current state till `end_time`

        Args:
            end_time (float): end time of the simulation.
        """
        if end_time < self.state.current_time:
            raise ValueError(
                f"End time {end_time} before start {self.state.current_time}"
            )

        # total progress bar must be a bit over so that the loop doesn't exceed total
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=tqdm.TqdmWarning)
            with tqdm.tqdm(
                total=end_time - self.state.current_time + self._delta_time,
                disable=not self.verbose,
            ) as progress_bar:

                while self.state.current_time + self._delta_time <= end_time:
                    self.state.current_time += self._delta_time
                    progress_bar.update(self._delta_time)
                    type(self).advance_state(self.state, self.debug)
                    self.state._previous_delta_time = self._delta_time
                    
