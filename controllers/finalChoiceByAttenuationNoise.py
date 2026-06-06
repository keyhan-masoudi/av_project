import math
import random

import numpy as np
from scipy.special import erfc

from NoiseConfigs.utilsFunctions import UtilsFunc
from config import Config
from models.node.cloud import CloudNode
from task_and_user_generator import Config as Cnf
from models.node.fog import FixedFogNode, FogLayerABC
from models.node.base import MobileNodeABC, green_bg, blue_bg


def red_bg(text):
    return f"\033[41m{text}\033[0m"


class FinalChoiceByAttenuationNoise:
    # THRESHOLD = 0.0001
    THRESHOLD = Config.NoiseConfig.DEFAULT_THRESHOLD
    # THRESHOLD = 50

    def makeFinalChoice(self, attenuationList, task, partitions, method):
        if len(attenuationList) > 0:
            if method == Config.NoiseMethod.FIRST_CHOICE:
                return self.firstChoiceMethod(attenuationList, task, partitions)
            elif method == Config.NoiseMethod.RANDOM_CHOICE:
                return self.randomMethod(attenuationList, task, partitions)
            elif method == Config.NoiseMethod.MIN_DISTANCE:
                return self.minDistanceMethod(attenuationList, task, partitions)

    def firstChoiceMethod(self, attenuationList, task, partitions):
        if attenuationList[0][2] != 0:
            plr_result, SNR, Pr, avgNoise, ber_result, snr_linear = self.calcPlr(attenuationList, attenuationList[0][2], task, partitions)
            task.SNR = snr_linear
            # print(red_bg(
            #     f"Pr:{Pr}, attenuation: {attenuationList[0][2]}, avgNoise:{avgNoise}, SNR:{SNR}, ber_result:{ber_result}, plr_result:{plr_result}"))
            return attenuationList[0], plr_result
        else:
            task.SNR = 0
            return attenuationList[0], 0

    def areAttenuationsZero(self, attenuationList):
        isThereAnyNonZero = False
        for i in range(0, len(attenuationList)):
            if attenuationList[i][1] != 0:
                isThereAnyNonZero = True

        return not isThereAnyNonZero

    def randomMethod(self, attenuationList, task, partitions):
        valid_entries = self.calcValidEntries(attenuationList, task, partitions)
        # print(f"valid_entries:{valid_entries}")
        finalChoice, snr_linear = random.choice(valid_entries)
        # print(green_bg(f"{finalChoice}, {snr_linear}"))
        task.SNR = snr_linear

        return finalChoice

    def calcValidEntries(self, attenuationList, task, partitions):
        valid_entries = []

        for i in range(0, len(attenuationList)):
            if attenuationList[i][2] != 0:
                plr_result, SNR, Pr, avgNoise, ber_result, snr_linear = self.calcPlr(attenuationList, attenuationList[i][2], task,
                                                                         partitions)

                valid_entries.append(((attenuationList[i], plr_result), snr_linear))
            else:
                valid_entries.append(((attenuationList[i], 0), 0))

        return valid_entries

    def calcSNR(self, attenuationList, attenuation, task, partitions):
        Pr = Cnf().VehicleConfig().HIGH_TRANSMISSION_POWER - attenuation + Config.AntennaGain.TX + Config.AntennaGain.RX
        avgNoise = FinalChoiceByAttenuationNoise().calcAvgNoise(attenuationList, task, partitions)

        SNR = Pr - avgNoise
        # print(red_bg(f"Pr: {Pr}, avgNoise: {avgNoise}, SNR: {SNR}"))

        return SNR, Pr, avgNoise

    def calcPlr(self, attenuationList, attenuation, task, partitions):

        SNR, Pr, avgNoise = self.calcSNR(attenuationList, attenuation, task, partitions)
        ber_result, snr_linear = FinalChoiceByAttenuationNoise().calculate_ber_bpsk(SNR)
        plr_result = FinalChoiceByAttenuationNoise().calculate_packet_loss(ber_result, 100)

        # print(red_bg(
        #         f"creator: ({task.creator.x}, {task.creator.y}), executor: ({attenuationList[0][1].x}, {attenuationList[0][1].y}), attenuation: {attenuationList[0][2]}, Pr:{Pr}, avgNoise:{avgNoise}, SNR:{SNR}, ber_result:{ber_result}, plr_result:{plr_result}"))

        return plr_result, SNR, Pr, avgNoise, ber_result, snr_linear

    def checkThreshold(self, attenuationList, task, partitions, considerDistance):
        valid_entries = []
        not_valids = []

        # print(red_bg(attenuationList))
        # print("---------------------------------------------------------------------------------------------------------")
        for i in range(0, len(attenuationList)):
            if attenuationList[i][2] != 0:
                # TRANSMISSION_LIST = Cnf.VehicleConfig.TRANSMISSION_LIST
                # for j in range(0, len(TRANSMISSION_LIST)):

                plr_result, SNR, Pr, avgNoise, ber_result, snr_linear = self.calcPlr(attenuationList, attenuationList[i][2], task,
                                                                         partitions)
                # if plr_result > FinalChoiceByAttenuationNoise().THRESHOLD:
                # print(red_bg(
                #         f"creator: ({task.creator.x}, {task.creator.y}), executor: ({attenuationList[i][1].x}, {attenuationList[i][1].y}), attenuation: {attenuationList[i][2]}, Pr:{Pr}, avgNoise:{avgNoise}, SNR:{SNR}, ber_result:{ber_result}, plr_result:{plr_result}"))
                if plr_result < Config.NoiseConfig.DEFAULT_THRESHOLD:
                    valid_entries.append((attenuationList[i], plr_result, snr_linear))
                    # break
                else:
                    not_valids.append((attenuationList[i], plr_result))
        # print("---------------------------------------------------------------------------------------------------------")

        if valid_entries:
            if considerDistance:
                # print("teeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeestttttttttttttttttttttttttttttttttt")
                best_entry = self.findMinValidEntries(valid_entries)
                task.SNR = best_entry[2]
                return best_entry[0], best_entry[1]

            else:
                # print(red_bg(f"valid_entries {valid_entries}"))
                best_entry = min(valid_entries, key=lambda x: x[1])
                # print(f"best_entry: {best_entry}")
                task.SNR = best_entry[2]
                # print(blue_bg(f"plr: {best_entry[1]}"))

                return best_entry[0], best_entry[1]
        else:
            for i in range(0, len(attenuationList)):
                if attenuationList[i][2] == 0:
                    task.SNR = 0
                    return attenuationList[i], 0
            # print(red_bg(f"not_valids : {not_valids}"))
            task.SNR = 0
            return None, None

    def checkThreshold2(self, attenuationList, task, partitions):
        valid_entries = []
        not_valids = []

        # print(red_bg(attenuationList))
        for i in range(0, len(attenuationList)):
            if attenuationList[i][2] != 0:
                # TRANSMISSION_LIST = Cnf.VehicleConfig.TRANSMISSION_LIST
                # for j in range(0, len(TRANSMISSION_LIST)):
                plr_result, SNR, Pr, avgNoise, ber_result, snr_linear = self.calcPlr(attenuationList, attenuationList[i][2], task,
                                                                         partitions)
                # if plr_result > FinalChoiceByAttenuationNoise().THRESHOLD:
                #     print(red_bg(f"Pr:{Pr}, attenuation: {attenuationList[i][2]}, avgNoise:{avgNoise}, SNR:{SNR}, ber_result:{ber_result}, plr_result:{plr_result}"))
                if plr_result < Config.NoiseConfig.DEFAULT_THRESHOLD:
                    valid_entries.append((attenuationList[i], plr_result, snr_linear))
                    # break
                else:
                    not_valids.append((attenuationList[i], plr_result))

        if valid_entries:
            best_entry = self.findBestDistanceValidEntries(task, valid_entries)
            task.SNR = best_entry[2]
            return best_entry[0], best_entry[1]

        else:
            for i in range(0, len(attenuationList)):
                if attenuationList[i][2] == 0:
                    task.SNR = 0
                    return attenuationList[i], 0
            return None, None

    def calculate_ber_bpsk(self, snr_db):
        """
        Calculate the Bit Error Rate (BER) for BPSK modulation given a single SNR in dB.

        Parameters:
        snr_db (float): Signal-to-Noise Ratio (SNR) in dB.

        Returns:
        float: BER value.
        """
        snr_linear = 10 ** (snr_db / 10)  # Convert SNR from dB to linear scale
        ber = 0.5 * erfc(np.sqrt(2 * snr_linear))  # BER formula for BPSK
        return ber, snr_linear

    def calculate_packet_loss(self, ber, packet_size):
        """Calculate packet loss probability given BER and packet size (in bits)."""
        plr = 1 - (1 - ber) ** packet_size  # Formula for Packet Loss Rate
        return plr * 100

    def calcAvgNoise(self, attenuationList, task, partitions) -> float:
        noises = []
        for i in range(0, len(attenuationList)):
            intersections = UtilsFunc().find_line_intersections(
                (task.creator.x, task.creator.y),
                (attenuationList[i][1].x, attenuationList[i][1].y), partitions)

            for k in range(0, len(intersections)):
                noises.append(intersections[k].trafficStatus.random_noise())

        finalNoise = 0
        for j in range(0, len(noises)):
            finalNoise += noises[j]
        finalNoise = finalNoise / len(noises)

        return finalNoise

    def minDistanceMethod(self, attenuationList, task, partitions):
        x = self.minDistanceItem(attenuationList)
        data = x[0]
        distance = x[1]

        # print(f"data : {data}, distance : {distance}")
        if data[2] != 0:
            plr_result, SNR, Pr, avgNoise, ber_result, snr_linear = self.calcPlr(attenuationList, data[2], task, partitions)
            task.SNR = snr_linear
            return data, plr_result
        else:
            task.SNR = 0
            return data, 0

    def calcMinDistance(self, zone, offloadedDevice):
        zoneX = zone.x
        zoneY = zone.y
        offloadedDeviceX = offloadedDevice.x
        offloadedDeviceY = offloadedDevice.y
        # print(red_bg(f"zoneX: {zoneX}, zoneY: {zoneY}, offloadedDeviceX: {offloadedDeviceX}, offloadedDeviceY: {offloadedDeviceY}"))
        distance = math.sqrt((offloadedDeviceX - zoneX) ** 2 + (offloadedDeviceY - zoneY) ** 2)
        return distance

    def minDistanceItem(self, attenuationList):
        distances = []
        if len(attenuationList) == 1 and not (attenuationList[0][0]):
            distances.append((attenuationList[0], 0))
            return distances[0]
        for i in range(0, len(attenuationList)):
            # print(green_bg(attenuationList[i]))
            distances.append(
                (attenuationList[i], self.calcMinDistance(attenuationList[i][0].zone, attenuationList[i][1])))
        min_distance = min(distances, key=lambda x: x[1])
        return min_distance

    def findMinValidEntries(self, valid_entries):
        distances = []
        print(red_bg(valid_entries))
        for i in range(0, len(valid_entries)):
            distances.append(
                (valid_entries[i][0], self.calcMinDistance(valid_entries[i][0][0][0].zone, valid_entries[i][0][0][1])))
            # print(red_bg(f"distances : {distances[i][1]}"))

        min_distance = min(distances, key=lambda x: x[1])
        return min_distance[0]

    def findBestDistanceValidEntries(self, task, valid_entries):
        finalEntries = []
        time = task.exec_time
        creator_next_position_x = (task.creator.x + task.creator.speed * time * np.cos(np.deg2rad(task.creator.angle)))
        creator_next_position_y = (task.creator.y + task.creator.speed * time * np.sin(np.deg2rad(task.creator.angle)))
        # print("Valid entries:", valid_entries)

        if valid_entries:
            for i in range(0, len(valid_entries)):
                # print("Valid entry:", valid_entries[i])
                if isinstance(valid_entries[i][0][1], FixedFogNode) or isinstance(valid_entries[i][0][1], CloudNode):
                    executor_next_position_x = valid_entries[i][0][1].x
                    executor_next_position_y = valid_entries[i][0][1].y
                else:
                    executor_next_position_x = (valid_entries[i][0][1].x + valid_entries[i][0][1].speed * time * np.cos(
                        np.deg2rad(valid_entries[i][0][1].angle)))
                    executor_next_position_y = (valid_entries[i][0][1].y + valid_entries[i][0][1].speed * time * np.sin(
                        np.deg2rad(valid_entries[i][0][1].angle)))

                finalEntries.append((valid_entries[i], np.sqrt(
                    (creator_next_position_y - executor_next_position_y) ** 2 +
                    (creator_next_position_x - executor_next_position_x) ** 2
                )))
        min_distance = min(finalEntries, key=lambda x: x[1])
        return min_distance[0]


if __name__ == "__main__":
    # Example usage:
    snr_input = float(input("Enter SNR (in dB): "))  # User inputs SNR value
    ber_result, snr_linear = FinalChoiceByAttenuationNoise().calculate_ber_bpsk(snr_input)
    plr_result = FinalChoiceByAttenuationNoise().calculate_packet_loss(ber_result, 100)

    print(f"SNR = {snr_input} dB → BER = {ber_result:.16f}")
    print(f"PLR = {plr_result:.16f}%")
