from hdf5_dataclass import HDF5Dataclass

from endgame_simulations.models import BaseInitialParams
from endgame_simulations.simulations import BaseState, GenericSimulation


class Params(BaseInitialParams):
    w_rate: float = 0.1
    delta_time: float = 0.2


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

class NewSim(
    GenericSimulation[Params, State], state_class=State, advance_state=advance_state
):
    @property
    def _delta_time(self) -> float:
        return self.state.params.delta_time


sim = NewSim(start_time=1, params=Params(w_rate=0.2))
sim.run(end_time=4)
