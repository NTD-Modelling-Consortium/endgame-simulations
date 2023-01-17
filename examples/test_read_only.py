from endgame_models import BaseInitialParams, get_read_only_differences, read_only


class SubTest(BaseInitialParams):
    z: int


class Test(BaseInitialParams):
    x: int
    y: int = read_only()


old = Test(x=1, y=2)
new = Test(x=1, y=3)

for i in get_read_only_differences(old, new):
    print(i)
