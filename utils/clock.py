class Clock:
    __time = 0.0

    @classmethod
    def set_current_time(cls, time):
        cls.__time = time

    @classmethod
    def get_current_time(cls) -> float:
        return cls.__time

    @classmethod
    def tick(cls):
        cls.__time += 1
