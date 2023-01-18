from abc import abstractproperty
from typing import ClassVar, Generic, Iterator, Protocol, TypeVar, cast, overload

import tqdm
from hdf5_dataclass import FileType

from endgame_simulations.models import (
    BaseInitialParams,
    BaseProgramParams,
    EndgameModel,
    _BaseUpdateParams,
)

from .common import AdvanceState, BaseState, State

ProgramModel = TypeVar("ProgramModel", bound=BaseProgramParams)
InitialParamsModel = TypeVar(
    "InitialParamsModel", bound=BaseInitialParams, contravariant=True
)
UpdateParamsModel = TypeVar(
    "UpdateParamsModel", bound=_BaseUpdateParams, contravariant=True
)
CombinedParams = TypeVar("CombinedParams", bound=BaseInitialParams, covariant=True)


class CombineParams(
    Protocol, Generic[InitialParamsModel, UpdateParamsModel, CombinedParams]
):
    def __call__(
        self, params: InitialParamsModel, update_params: UpdateParamsModel
    ) -> CombinedParams:
        ...


class GenericEndgame(
    Generic[InitialParamsModel, UpdateParamsModel, ProgramModel, State, CombinedParams]
):
    state_class: ClassVar[type[BaseState]]
    combine_params: ClassVar[CombineParams]
    advance_state: ClassVar[AdvanceState]
    state: State
    verbose: bool
    debug: bool
    _param_set: list[CombinedParams]

    def __init_subclass__(
        cls,
        *,
        state_class: type[State],
        advance_state: AdvanceState,
        combine_params: CombineParams,
    ) -> None:
        cls.state_class = state_class
        cls.advance_state = advance_state
        cls.combine_params = combine_params

    def __init__(
        self,
        *,
        start_time: float | None = None,
        endgame: EndgameModel[InitialParamsModel, UpdateParamsModel, ProgramModel]
        | None = None,
        input: FileType | None = None,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        assert (endgame is not None) != (
            input is not None
        ), "You must provide either `endgame` or `input`"
        if endgame:
            state = self.state_class.from_params(endgame, start_time or 0.0)

        else:
            # input
            state = self.state_class.from_hdf5(input)
        self.state = cast(State, state)
        self.verbose = verbose
        self.debug = debug

    @abstractproperty
    def _delta_time(self) -> float:
        ...

    def save(self, output: FileType) -> None:
        """Save the simulation to a file/stream.

        The output file will be in a HDF5 format. The simulation can then be
        restored with `Simulation.restore` class method.

        Args:
            output (FileType): output file/stream
        """
        self.state.to_hdf5(output)

    @classmethod
    def restore(cls, input: FileType):
        """Restore the simulation from a file/stream

        Args:
            input (FileType): HDF5 stream/file

        Returns:
            Simulation: restored simulation
        """
        return cls(input=input)

    @overload
    def iter_run(self, *, end_time: float, sampling_interval: float) -> Iterator[State]:
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

        Yields:
            Iterator[State]: Iterator of the simulation's state.
        """
        ...

    @overload
    def iter_run(
        self, *, end_time: float, sampling_years: list[float]
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

        Yields:
            Iterator[State]: Iterator of the simulation's state.
        """
        ...

    def iter_run(
        self,
        *,
        end_time: float,
        sampling_interval: float | None = None,
        sampling_years: list[float] | None = None,
    ) -> Iterator[State]:
        if end_time < self.state.current_time:
            raise ValueError("End time after start")

        if sampling_interval and sampling_years:
            raise ValueError(
                "You must provide sampling_interval, sampling_years or neither"
            )

        if sampling_years:
            sampling_years = sorted(sampling_years)

        sampling_years_idx = 0

        with tqdm.tqdm(
            total=end_time - self.state.current_time + self._delta_time,
            disable=not self.verbose,
        ) as progress_bar:
            while self.state.current_time <= end_time:
                is_on_sampling_interval = (
                    sampling_interval is not None
                    and self.state.current_time % sampling_interval < self._delta_time
                )

                is_on_sampling_year = (
                    sampling_years
                    and sampling_years_idx < len(sampling_years)
                    and abs(
                        self.state.current_time - sampling_years[sampling_years_idx]
                    )
                    < self._delta_time
                )

                if is_on_sampling_interval or is_on_sampling_year:
                    yield self.state
                    if is_on_sampling_year:
                        sampling_years_idx += 1

                progress_bar.update(self._delta_time)
                type(self).advance_state(self.state, self.debug)

    def run(self, *, end_time: float) -> None:
        """Run simulation from current state till `end_time`

        Args:
            end_time (float): end time of the simulation.
        """
        if end_time < self.state.current_time:
            raise ValueError("End time after start")

        # total progress bar must be a bit over so that the loop doesn't exceed total
        with tqdm.tqdm(
            total=end_time - self.state.current_time + self._delta_time,
            disable=not self.verbose,
        ) as progress_bar:
            while self.state.current_time <= end_time:
                progress_bar.update(self._delta_time)
                type(self).advance_state(self.state)