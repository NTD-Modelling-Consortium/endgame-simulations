import json
from typing import ClassVar, Generic, Iterator, Protocol, TypeVar, cast, overload

import h5py
from hdf5_dataclass import FileType

from endgame_simulations.models import BaseInitialParams, EndgameModel
from endgame_simulations.simulations import GenericSimulation

from .common import State

CombinedParams = TypeVar("CombinedParams", bound=BaseInitialParams)
EndgameModelGeneric = TypeVar(
    "EndgameModelGeneric", bound=EndgameModel, contravariant=True
)


class ConvertEndgame(Protocol, Generic[EndgameModelGeneric, CombinedParams]):
    def __call__(
        self, endgame: EndgameModelGeneric
    ) -> list[tuple[float, CombinedParams]]:
        ...


Simulation = TypeVar("Simulation", bound=GenericSimulation)


class GenericEndgame(Generic[EndgameModelGeneric, Simulation, State, CombinedParams]):
    simulation_class: ClassVar[type[GenericSimulation]]
    combined_params_model: ClassVar[type[BaseInitialParams]]
    convert_endgame: ClassVar[ConvertEndgame]
    simulation: Simulation
    _param_set: list[tuple[float, CombinedParams]]
    next_params_index: int

    def __init_subclass__(
        cls,
        *,
        simulation_class: type[Simulation],
        convert_endgame: ConvertEndgame,
        combined_params_model: type[CombinedParams],
    ) -> None:
        cls.simulation_class = simulation_class
        cls.convert_endgame = convert_endgame
        cls.combined_params_model = combined_params_model

    @overload
    def __init__(
        self,
        *,
        start_time: float,
        endgame: EndgameModelGeneric,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Create a new endgame simulation, given the parameters.

        Args:
            start_time (float): Start time of the simulation
            endgame (EndgameModelGeneric): The endgame model description.
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
        """Restore the endgame simulation from a previously saved file.

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
        endgame: EndgameModelGeneric | None = None,
        input: FileType | h5py.File | h5py.Group | None = None,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        assert (endgame is not None) != (
            input is not None
        ), "You must provide either `endgame` or `input`"

        if endgame:
            self._param_set = type(self).convert_endgame(endgame)
            assert start_time is not None and (len(self._param_set) > 0)
            time_params = [t for t, _ in self._param_set]
            self.next_params_index = next(
                i for i, value in enumerate(time_params) if value > start_time
            )
            if self.next_params_index < 1:
                raise ValueError(f"Invalid next param index: {self.next_params_index}")
            simulation = type(self).simulation_class(
                start_time=start_time,
                params=self._param_set[self.next_params_index - 1][1],
                verbose=verbose,
                debug=debug,
            )

        else:
            assert input
            if isinstance(input, (h5py.File, h5py.Group)):
                h5 = input
                sim = h5["simulation"]
                param_set_str = h5.attrs["param_set"]
                next_params = h5.attrs["next_params_index"]
                assert isinstance(sim, h5py.Group)
                simulation = type(self).simulation_class.restore(input=sim)
            else:
                with h5py.File(input, "r") as h5:
                    sim = h5["simulation"]
                    param_set_str = h5.attrs["param_set"]
                    next_params = h5.attrs["next_params_index"]
                    assert isinstance(sim, h5py.Group)
                    simulation = type(self).simulation_class.restore(input=sim)

            assert isinstance(param_set_str, str)
            param_set: list[tuple[float, dict]] = json.loads(param_set_str)
            converted_param_set = [
                (i[0], self.combined_params_model.parse_obj(i[1])) for i in param_set
            ]
            self._param_set = cast(
                list[tuple[float, CombinedParams]], converted_param_set
            )

            assert not isinstance(next_params, h5py.Empty)
            self.next_params_index = int(next_params)

        self.simulation = cast(Simulation, simulation)

    def reset_endgame(self, endgame: EndgameModelGeneric):
        self._param_set = type(self).convert_endgame(endgame)
        assert len(self._param_set) > 0
        time_params = [t for t, _ in self._param_set]
        self.next_params_index = next(
            i
            for i, value in enumerate(time_params)
            if value > self.simulation.state.current_time
        )
        if self.next_params_index < 1:
            raise ValueError(f"Invalid next param index: {self.next_params_index}")
        self.simulation.reset_current_params(
            self._param_set[self.next_params_index - 1][1]
        )

    def save(self, output: FileType | h5py.File | h5py.Group) -> None:
        """Save the simulation to a file/stream.

        The output file will be in a HDF5 format. The simulation can then be
        restored with `Simulation.restore` class method.

        Args:
            output (FileType): output file/stream
        """
        if isinstance(output, (h5py.File, h5py.Group)):
            h5 = output
            grp = h5.create_group("simulation")
            self.simulation.save(grp)
            h5.attrs["param_set"] = json.dumps(
                [(i[0], i[1].dict()) for i in self._param_set]
            )
            h5.attrs["next_params_index"] = self.next_params_index
        else:
            with h5py.File(output, "w") as h5:
                grp = h5.create_group("simulation")
                self.simulation.save(grp)
                h5.attrs["param_set"] = json.dumps(
                    [(i[0], i[1].dict()) for i in self._param_set]
                )
                h5.attrs["next_params_index"] = self.next_params_index

    @classmethod
    def restore(cls, input: FileType | h5py.File | h5py.Group):
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
            checkpoint_time = end_time
            if self.next_params_index < len(self._param_set):
                # Has param sets left
                time, params = self._param_set[self.next_params_index]
                self.next_params_index += 1
                if time < end_time:
                    checkpoint_time = time
                    self.simulation.reset_current_params(params)

            yield from self.simulation.iter_run(
                end_time=checkpoint_time,
                sampling_interval=sampling_interval,
                sampling_years=sampling_years,
            )

    def run(self, *, end_time: float) -> None:
        """Run simulation from current state till `end_time`

        Args:
            end_time (float): end time of the simulation.
        """
        while self.simulation.state.current_time < end_time:
            checkpoint_time = end_time
            if self.next_params_index < len(self._param_set):
                # Has param sets left
                time, params = self._param_set[self.next_params_index]
                self.next_params_index += 1
                if time < end_time:
                    checkpoint_time = time
                    self.simulation.reset_current_params(params)
            self.simulation.run(end_time=checkpoint_time)
