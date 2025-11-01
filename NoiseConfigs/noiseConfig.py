import random
from config import Config


class NoiseConfig:
    class TrafficNoiseConfig:
        MAX_NOISE: float
        Q3_NOISE: float
        MEAN_NOISE: float
        Q1_NOISE: float
        MIN_NOISE: float

        def __init__(self, max_noise=None, q3_noise=None, mean_noise=None, q1_noise=None, min_noise=None):
            self.MAX_NOISE = max_noise
            self.Q3_NOISE = q3_noise
            self.MEAN_NOISE = mean_noise
            self.Q1_NOISE = q1_noise
            self.MIN_NOISE = min_noise

        def random_noise(self) -> float:
            # produce random number with gaussian distribution
            sigma = (self.Q3_NOISE - self.Q1_NOISE) / 1.35  # 1.35 due to IQR
            random_number = random.gauss(mu=self.MEAN_NOISE, sigma=sigma)
            return random_number-200

    class GreenTraffic(TrafficNoiseConfig):
        def __init__(self):
            super().__init__(max_noise=Config.TrafficNoise.GreenTrafficNoise.DEFAULT_GreenTrafficNoise[0], q3_noise=Config.TrafficNoise.GreenTrafficNoise.DEFAULT_GreenTrafficNoise[1], mean_noise=Config.TrafficNoise.GreenTrafficNoise.DEFAULT_GreenTrafficNoise[2], q1_noise=Config.TrafficNoise.GreenTrafficNoise.DEFAULT_GreenTrafficNoise[3], min_noise=Config.TrafficNoise.GreenTrafficNoise.DEFAULT_GreenTrafficNoise[4])

    class YellowTraffic(TrafficNoiseConfig):
        def __init__(self):
            super().__init__(max_noise=Config.TrafficNoise.YellowTrafficNoise.DEFAULT_YellowTrafficNoise[0], q3_noise=Config.TrafficNoise.YellowTrafficNoise.DEFAULT_YellowTrafficNoise[1], mean_noise=Config.TrafficNoise.YellowTrafficNoise.DEFAULT_YellowTrafficNoise[2], q1_noise=Config.TrafficNoise.YellowTrafficNoise.DEFAULT_YellowTrafficNoise[3], min_noise=Config.TrafficNoise.YellowTrafficNoise.DEFAULT_YellowTrafficNoise[4])

    class OrangeTraffic(TrafficNoiseConfig):
        def __init__(self):
            super().__init__(max_noise=Config.TrafficNoise.OrangeTrafficNoise.DEFAULT_OrangeTrafficNoise[0], q3_noise=Config.TrafficNoise.OrangeTrafficNoise.DEFAULT_OrangeTrafficNoise[1], mean_noise=Config.TrafficNoise.OrangeTrafficNoise.DEFAULT_OrangeTrafficNoise[2], q1_noise=Config.TrafficNoise.OrangeTrafficNoise.DEFAULT_OrangeTrafficNoise[3], min_noise=Config.TrafficNoise.OrangeTrafficNoise.DEFAULT_OrangeTrafficNoise[4])

    class RedTraffic(TrafficNoiseConfig):
        def __init__(self):
            super().__init__(max_noise=Config.TrafficNoise.RedTrafficNoise.DEFAULT_RedTrafficNoise[0], q3_noise=Config.TrafficNoise.RedTrafficNoise.DEFAULT_RedTrafficNoise[1], mean_noise=Config.TrafficNoise.RedTrafficNoise.DEFAULT_RedTrafficNoise[2], q1_noise=Config.TrafficNoise.RedTrafficNoise.DEFAULT_RedTrafficNoise[3], min_noise=Config.TrafficNoise.RedTrafficNoise.DEFAULT_RedTrafficNoise[4])

    class BlackTraffic(TrafficNoiseConfig):
        def __init__(self):
            super().__init__(max_noise=Config.TrafficNoise.BlackTrafficNoise.DEFAULT_BlackTrafficNoise[0], q3_noise=Config.TrafficNoise.BlackTrafficNoise.DEFAULT_BlackTrafficNoise[1], mean_noise=Config.TrafficNoise.BlackTrafficNoise.DEFAULT_BlackTrafficNoise[2], q1_noise=Config.TrafficNoise.BlackTrafficNoise.DEFAULT_BlackTrafficNoise[3], min_noise=Config.TrafficNoise.BlackTrafficNoise.DEFAULT_BlackTrafficNoise[4])

    # parent class Urban
    class UrbanNoiseConfig:
        variance: float = (1.366 - 1.239) / 1.35  # different of mean (Q1, Q3) of real data /IQR
        baseCoff: float

        def __init__(self, baseCoff):
            self.baseCoff = baseCoff

        def random_noise_coff(self) -> float:
            # produce random number with gaussian distribution
            sigma = self.variance
            random_number = random.gauss(mu=self.baseCoff, sigma=sigma)
            return random_number

    class Rural(UrbanNoiseConfig):
        def __init__(self):
            super().__init__(Config.AttenuationLevel.DEFAULT_AttenuationLevel[0])

    class GoodUrban(UrbanNoiseConfig):
        def __init__(self):
            super().__init__(Config.AttenuationLevel.DEFAULT_AttenuationLevel[1])

    class MediumUrban(UrbanNoiseConfig):
        def __init__(self):
            super().__init__(Config.AttenuationLevel.DEFAULT_AttenuationLevel[2])

    class BadUrban(UrbanNoiseConfig):
        def __init__(self):
            super().__init__(Config.AttenuationLevel.DEFAULT_AttenuationLevel[3])

    class RainNoiseConfig:
        baseAttenuation: float = 0
        variance: float = 0.5  # selected by my observation and idea

        def __init__(self, baseAttenuation, variance=variance):
            self.baseAttenuation = baseAttenuation
            self.variance = variance

        def random_noise(self) -> float:
            # produce random number with gaussian distribution
            sigma = self.variance
            random_number = random.gauss(mu=self.baseAttenuation, sigma=sigma)
            return random_number

    class Rain0(RainNoiseConfig):
        def __init__(self):
            super().__init__(0, 0.2)

        #  override random noise for this condition
        def random_noise(self) -> float:
            random_number = super().random_noise()
            if random_number < 0:
                return 0
            else:
                return random_number

    class Rain13(RainNoiseConfig):
        def __init__(self):
            super().__init__(baseAttenuation=1)

    class Rain23(RainNoiseConfig):
        def __init__(self):
            super().__init__(baseAttenuation=2)

    class Rain50(RainNoiseConfig):
        def __init__(self):
            super().__init__(baseAttenuation=4)

    class Rain100(RainNoiseConfig):
        def __init__(self):
            super().__init__(baseAttenuation=5)

    class Rain150(RainNoiseConfig):
        def __init__(self):
            super().__init__(baseAttenuation=7)

    class Rain200(RainNoiseConfig):
        def __init__(self):
            super().__init__(baseAttenuation=9)


# test functions
if __name__ == "__main__":
    # test Rain Noise
    Rain0 = NoiseConfig.Rain0()
    Rain13 = NoiseConfig.Rain13()
    Rain23 = NoiseConfig.Rain23()
    Rain50 = NoiseConfig.Rain50()
    Rain100 = NoiseConfig.Rain100()
    Rain150 = NoiseConfig.Rain150()
    Rain200 = NoiseConfig.Rain200()

    print(f"Rain Noise0: {Rain0.random_noise()} ")
    print(f"Rain Noise13: {Rain13.random_noise()} ")
    print(f"Rain Noise23: {Rain23.random_noise()} ")
    print(f"Rain Noise50: {Rain50.random_noise()} ")
    print(f"Rain Noise100: {Rain100.random_noise()} ")
    print(f"Rain Noise150: {Rain150.random_noise()} ")
    print(f"Rain Noise200: {Rain200.random_noise()} \n")
    x = 0
    y1 = 0
    miny1=13
    maxY1=0
    for i in range(0, 100):
        x += Rain0.random_noise()
        z = Rain50.random_noise()
        y1 += z
        if z < miny1:
            miny1 = z
        if z > maxY1:
            maxY1 = z

    print(f"avg Rain Noise0 : {x / 100}")
    print(f"avg Rain Noise50 : {y1 / 100}")
    print(f"min Rain Noise50 : {miny1}")
    print(f"max Rain Noise50 : {maxY1}\n")

    # test Urban Noise
    rural = NoiseConfig.Rural()
    goodUrban = NoiseConfig.GoodUrban()
    mediumUrban = NoiseConfig.MediumUrban()
    badUrban = NoiseConfig.BadUrban()

    print(f"rural random noise coff: {rural.random_noise_coff()} ")
    print(f"goodUrban random noise coff: {goodUrban.random_noise_coff()} ")
    print(f"mediumUrban random noise coff: {mediumUrban.random_noise_coff()} ")
    print(f"badUrban random noise coff: {badUrban.random_noise_coff()} \n")
    x = 0
    y1 = 0
    y2 = 0
    y3 = 0
    for i in range(0, 100):
        x += rural.random_noise_coff()
        y1 += goodUrban.random_noise_coff()
        y2 += mediumUrban.random_noise_coff()
        y3 += badUrban.random_noise_coff()
    print(f"avg rural coff : {x / 100}")
    print(f"avg goodUrban coff : {y1 / 100}")
    print(f"avg mediumUrban coff : {y2 / 100}")
    print(f"avg badUrban coff : {y3 / 100}\n")

    # test Traffic Noise
    black_traffic = NoiseConfig.BlackTraffic()
    red_traffic = NoiseConfig.RedTraffic()
    orange_traffic = NoiseConfig.OrangeTraffic()
    green_traffic = NoiseConfig.GreenTraffic()
    # x = 0
    # l1 = 0
    # l2 = 0
    # for i in range(0, 100):
    #     x = black_traffic.random_noise()
    #     if x > 65 or x < 45:
    #         l1 += 1
    #     if 60 > x > 50:
    #         l2 += 1

    # print(f"l1:{l1}, l2:{l2}")
    print("Black Traffic Random Noise:", black_traffic.random_noise())
    print("Red Traffic Random Noise:", red_traffic.random_noise())
    print("Orange Traffic Random Noise:", orange_traffic.random_noise())
    print("Green Traffic Random Noise:", green_traffic.random_noise())