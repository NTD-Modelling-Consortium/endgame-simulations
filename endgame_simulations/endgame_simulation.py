from abc import abstractproperty
from typing import ClassVar, Generic, Iterator, Protocol, TypeVar, cast, overload

from hdf5_dataclass import FileType

from endgame_simulations.models import BaseInitialParams, EndgameModel
from endgame_simulations.simulations import GenericSimulation

from .common import AdvanceState, State

CombinedParams = TypeVar("CombinedParams", bound=BaseInitialParams)
EndgameModelGeneric = TypeVar(
    "EndgameModelGeneric", bound=EndgameModel, contravariant=True
)


class ConvertEndgame(Protocol, Generic[EndgameModelGeneric, CombinedParams]):
    def __call__(self, endgame: EndgameModelGeneric) -> list[CombinedParams]:
        ...


Simulation = TypeVar("Simulation", bound=GenericSimulation)


class GenericEndgame(Generic[EndgameModelGeneric, Simulation, State, CombinedParams]):
    simulation_class: ClassVar[type[GenericSimulation]]
    convert_endgame: ClassVar[ConvertEndgame]
    advance_state: ClassVar[AdvanceState]
    simulation: Simulation
    verbose: bool
    debug: bool
    _param_set: list[tuple[float, CombinedParams]]
    current_param: int

    def __init_subclass__(
        cls,
        *,
        simulation_class: type[Simulation],
        advance_state: AdvanceState,
        convert_endgame: ConvertEndgame,
    ) -> None:
        cls.simulation_class = simulation_class
        cls.advance_state = advance_state
        cls.convert_endgame = convert_endgame

    def __init__(
        self,
        *,
        start_time: float | None = None,
        endgame: EndgameModelGeneric | None = None,
        input: FileType | None = None,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        assert (endgame is not None) != (
            input is not None
        ), "You must provide either `endgame` or `input`"

        if endgame:
            self._param_set = type(self).convert_endgame(endgame)
            assert start_time
            simulation = type(self).simulation_class(
                start_time=start_time,
                params=self._param_set[0],
                verbose=verbose,
                debug=debug,
            )

        else:
            assert input
            # input
            simulation = type(self).simulation_class.restore(input=input)
        self.simulation = cast(Simulation, simulation)
        self.verbose = verbose
        self.debug = debug
        self.current_param = 1

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
        self.simulation.save(output)

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
        while self.simulation.state.current_time < end_time:
            if self.current_param < len(self._param_set):
                # Has param sets left
                time, params = self._param_set[self.current_param]
                if time < end_time:
                    self.simulation.reset_current_params(params)
                    yield next(
                        self.simulation.iter_run(
                            end_time=time,
                            sampling_interval=sampling_interval,
                            sampling_years=sampling_years,
                        )
                    )
                else:
                    yield next(
                        self.simulation.iter_run(
                            end_time=end_time,
                            sampling_interval=sampling_interval,
                            sampling_years=sampling_years,
                        )
                    )
            else:
                yield next(
                    self.simulation.iter_run(
                        end_time=end_time,
                        sampling_interval=sampling_interval,
                        sampling_years=sampling_years,
                    )
                )

    def run(self, *, end_time: float) -> None:
        """Run simulation from current state till `end_time`

        Args:
            end_time (float): end time of the simulation.
        """
        while self.simulation.state.current_time < end_time:
            if self.current_param < len(self._param_set):
                # Has param sets left
                time, params = self._param_set[self.current_param]
                if time < end_time:
                    self.simulation.reset_current_params(params)
                    self.simulation.run(end_time=time)
                else:
                    self.simulation.run(end_time=end_time)
            else:
                self.simulation.run(end_time=end_time)
