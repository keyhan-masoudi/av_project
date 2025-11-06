class Config:
    CHUNK_SIZE = 3600
    NEGATIVE_REWARD = -10

    class SimulatorConfig:
        SIMULATION_DURATION = 3600
        TIMEOUT_TIME = 2
        BANDWIDTH = 150

    class CloudConfig:
        # todo: add number of core
        DEFAULT_X = 6000
        DEFAULT_Y = 1500
        DEFAULT_RADIUS = 10000
        CLOUD_BANDWIDTH = 60
        MAX_TASK_QUEUE_LEN = 2000
        DEFAULT_COMPUTATION_POWER = 3500
        CLOUD_NODE_FREQUENCY = 5
        POWER_LIMIT = 0.99
        NUM_CORE = 10

    class FixedFogNodeConfig:
        # todo: add number of core
        MAX_TASK_QUEUE_LEN = 400
        DEFAULT_COMPUTATION_POWER = 500
        Fixed_NODE_FREQUENCY = 2
        POWER_LIMIT = 0.9
        NUM_CORE = 4

    class MobileFogNodeConfig:
        # todo: add number of core
        DEFAULT_RADIUS = 150
        MAX_TASK_QUEUE_LEN = 150
        DEFAULT_COMPUTATION_POWER = 200
        MOBILE_NODE_FREQUENCY = 1.5
        POWER_LIMIT = 0.6
        NUM_CORE = 1

    class UserNodeConfig:
        MAX_TASK_QUEUE_LEN = 10
        DEFAULT_COMPUTATION_POWER = 20
        USER_NODE_FREQUENCY = 0.5
        LOCAL_OFFLOAD_POWER_OVERHEAD = 1
        LOCAL_EXECUTE_TIME_OVERHEAD = 1
        POWER_LIMIT = 0.4
        NUM_CORE = 1

    class CriticalUserNodeConfig:
        MAX_TASK_QUEUE_LEN = 10
        DEFAULT_COMPUTATION_POWER = 20
        USER_NODE_FREQUENCY = 2
        LOCAL_OFFLOAD_POWER_OVERHEAD = 1
        LOCAL_EXECUTE_TIME_OVERHEAD = 1
        POWER_LIMIT = 0.4
        NUM_CORE = 1

    class ZoneManagerConfig:
        ALGORITHM_RANDOM = "Random"
        ALGORITHM_HEURISTIC = "Heuristic"
        ALGORITHM_HRL = "HRL"
        ALGORITHM_ONLY_CLOUD = "Only Cloud"
        ALGORITHM_ONLY_FOG = "Only Fog"
        ALGORITHM_DEEP_RL = "DeepRL"
        ALGORITHM_HEURISTIC2 = "Heuristic2"
        ALGORITHM_MADDPG = "MADDPG"
        ALGORITHM_DDPG = "DDPG"
        ALGORITHM_PPO = "PPO"
        ALGORITHM_SAC = "SAC"

        DEFAULT_ALGORITHM = ALGORITHM_RANDOM

    class NoiseMethod:
        PROPOSED_METHOD = "Proposed Method"
        PROPOSED_METHOD2 = "Proposed Method2"
        PROPOSED_METHOD3 = "Proposed Method3"
        FIRST_CHOICE = "First Choice"
        RANDOM_CHOICE = "Random Choice"
        MIN_DISTANCE = "Min Distance"

        DEFAULT_METHOD = FIRST_CHOICE

    class RandomZoneManagerConfig:
        OFFLOAD_CHANCE: float = 0.5

    class AntennaGain:
        TX: float = 27
        RX: float = -5

    class TaskConfig:
        # note: PACKET_COST_PER_METER = 0.001
        PACKET_COST_PER_METER = 0.001
        # PACKET_COST_PER_METER = 0.005

        # note: TASK_COST_PER_METER = 0.005
        # TASK_COST_PER_METER = 0.01
        TASK_COST_PER_METER = 0.005

        MIGRATION_OVERHEAD = 0.01
        CLOUD_PROCESSING_OVERHEAD = 0.5

    class NoiseConfig:
        T1 = 25
        T2 = 50
        T3 = 75
        NONE = -1

        DEFAULT_THRESHOLD = NONE

    class AttenuationLevel:
        AttenuationLevel1 = [2.3, 2.5, 2.7, 2.9]
        AttenuationLevel2 = [2.4, 2.7, 3, 3.3]
        AttenuationLevel1Name = 1
        AttenuationLevel2Name = 2

        DEFAULT_AttenuationLevel = AttenuationLevel1
        DEFAULT_AttenuationLevelName = AttenuationLevel1Name

    class TrafficNoise:
        TrafficNoiseLevel1 = 0
        TrafficNoiseLevel2 = 1
        DEFAULT_TrafficNoiseLevel = TrafficNoiseLevel1

        class GreenTrafficNoise:
            GreenTrafficNoise1 = [95, 91, 90, 89, 85]
            GreenTrafficNoise2 = [99, 95, 94, 93, 89]

            DEFAULT_GreenTrafficNoise = GreenTrafficNoise1

        class YellowTrafficNoise:
            YellowTrafficNoise1 = [100, 96, 95, 94, 90]
            YellowTrafficNoise2 = [102, 98, 97, 96, 92]

            DEFAULT_YellowTrafficNoise = YellowTrafficNoise1

        class OrangeTrafficNoise:
            OrangeTrafficNoise1 = [105, 101, 100, 99, 95]
            OrangeTrafficNoise2 = [105, 101, 100, 99, 95]

            DEFAULT_OrangeTrafficNoise = OrangeTrafficNoise1

        class RedTrafficNoise:
            RedTrafficNoise1 = [110, 106, 105, 104, 100]
            RedTrafficNoise2 = [108, 104, 103, 102, 98]

            DEFAULT_RedTrafficNoise = RedTrafficNoise1

        class BlackTrafficNoise:
            BlackTrafficNoise1 = [115, 111, 110, 109, 105]
            BlackTrafficNoise2 = [111, 107, 106, 105, 101]

            DEFAULT_BlackTrafficNoise = BlackTrafficNoise1