from hdf5_dataclass import HDF5Dataclass

from endgame_simulations.endgame_simulation import GenericEndgame
from endgame_simulations.models import (
    BaseInitialParams,
    BaseProgramParams,
    EndgameModel,
    make_endgame_model,
)
from endgame_simulations.simulations import BaseState, GenericSimulation


class Params(BaseInitialParams):
    w_rate: float = 0.1
    delta_time: float = 0.2


class TreatmentParams(BaseProgramParams):
    treatment_start: float = 0.0


class FullParams(Params):
    treatment: TreatmentParams = TreatmentParams(treatment_interval=0.1)


class State(HDF5Dataclass, BaseState[Params]):
    current_time: float
    params: Params
    state_int: int = 0

    @classmethod
    def from_params(cls, params: Params, current_time):
        return cls(params=params, current_time=current_time)

    def get_params(self) -> Params:
        return self.params

    def reset_params(self, params: Params):
        self.params = params


def advance_state(state: State, debug: bool = False):
    print(state)
    state.state_int = state.state_int + 1


class TestSimulation(
    GenericSimulation[Params, State], state_class=State, advance_state=advance_state
):
    @property
    def _delta_time(self) -> float:
        return self.state.params.delta_time


TestEndgame = make_endgame_model("TestEndgame", Params, TreatmentParams)
input_end = {
    "parameters": {
        "initial": {"w_rate": 0.1, "delta_time": 3},
        "changes": [
            {"year": 2020, "month": 1, "params": {"delta_time": 1}},
            {"year": 2022, "month": 1, "params": {"delta_time": 2}},
        ],
    },
    "programs": [],
}

endgame = TestEndgame.parse_obj(input_end)


def convert_endgame(endgame: EndgameModel) -> list[FullParams]:
    return [FullParams()]


class NewEndgame(
    GenericEndgame[TestEndgame, TestSimulation, State, FullParams],
    convert_endgame=convert_endgame,
    simulation_class=TestSimulation,
):
    @property
    def _delta_time(self) -> float:
        return self.simulation.state.params.delta_time


sim = NewEndgame(start_time=1, endgame=endgame)

sim.run(end_time=4)
