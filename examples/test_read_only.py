from endgame_models.get_differences import get_differences
from endgame_models.models import BaseInitialParams, read_only


class SubTest(BaseInitialParams):
    z: int


class Test(BaseInitialParams):
    x: int
    y: int = read_only()


old = Test(x=1, y=2)
new = Test(x=1, y=3)

for i in get_differences(old.dict(), new.dict(), Test):
    print(i)
