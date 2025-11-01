import math
import random

import numpy as np
from scipy.special import erfc

from NoiseConfigs.utilsFunctions import UtilsFunc
from config import Config
from task_and_user_generator import Config as Cnf


def red_bg(text):
    return f"\033[41m{text}\033[0m"


class FinalChoiceByAttenuationNoise:
    # THERESHOLD = 0.0001
    THERESHOLD = 20

    def makeFinalChoice(self, attenuationList, task, partitions, method):
        if len(attenuationList) > 0:

            if method == Config.NoiseMethod.PROPOSED_METHOD:
                # check if all values are 0 offload it locally
                if self.areAttenuationsZero(attenuationList):
                    return attenuationList[0], 0
                choiceByCheckingNoise = self.checkThreshold(attenuationList, task, partitions, considerDistance=False)
                # print(red_bg(self.checkThreshold(attenuationList, task, partitions)))
                return choiceByCheckingNoise

            elif method == Config.NoiseMethod.PROPOSED_METHOD2:
                if self.areAttenuationsZero(attenuationList):
                    return attenuationList[0], 0
                choiceByCheckingNoise = self.checkThreshold(attenuationList, task, partitions, considerDistance=True)
                # print(red_bg(self.checkThreshold(attenuationList, task, partitions)))
                return choiceByCheckingNoise

            elif method == Config.NoiseMethod.RANDOM_CHOICE:
                return self.randomMethod(attenuationList, task, partitions)

            elif method == Config.NoiseMethod.FIRST_CHOICE:
                return self.firstChoiceMethod(attenuationList, task, partitions)

            elif method == Config.NoiseMethod.MIN_DISTANCE:
                return self.minDistanceMethod(attenuationList, task, partitions)

    def firstChoiceMethod(self, attenuationList, task, partitions):
        if attenuationList[0][2] != 0:
            plr_result = self.calcPlr(attenuationList, attenuationList[0][2], task, partitions)
            return attenuationList[0], plr_result
        else:
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
        return random.choice(valid_entries)

    def calcValidEntries(self, attenuationList, task, partitions):
        valid_entries = []

        for i in range(0, len(attenuationList)):
            if attenuationList[i][2] != 0:
                plr_result = self.calcPlr(attenuationList, attenuationList[i][2], task, partitions)

                valid_entries.append((attenuationList[i], plr_result))
            else:
                valid_entries.append((attenuationList[i], 0))

        return valid_entries

    def calcSNR(self, attenuationList, attenuation, task, partitions):
        transmissionPower = Cnf.VehicleConfig.HIGH_TRANSMISSION_POWER

        # note : removed! it will be use in another project
        # if TransmissionType == Cnf.VehicleConfig.HIGH_TRANSMISSION_POWER:
        #     transmissionPower = Cnf.VehicleConfig.HIGH_TRANSMISSION_POWER
        # elif TransmissionType == Cnf.VehicleConfig.MEDIUM_TRANSMISSION_POWER:
        #     transmissionPower = Cnf.VehicleConfig.MEDIUM_TRANSMISSION_POWER
        # elif TransmissionType == Cnf.VehicleConfig.LOW_TRANSMISSION_POWER:
        #     transmissionPower = Cnf.VehicleConfig.LOW_TRANSMISSION_POWER

        Pr = transmissionPower - attenuation
        avgNoise = FinalChoiceByAttenuationNoise().calcAvgNoise(attenuationList, task, partitions)

        SNR = Pr - avgNoise
        print(red_bg(f"Pr: {Pr}, avgNoise: {avgNoise}, SNR: {SNR}"))
        return SNR

    def calcPlr(self, attenuationList, attenuation, task, partitions):

        SNR = self.calcSNR(attenuationList, attenuation, task, partitions)
        ber_result = FinalChoiceByAttenuationNoise().calculate_ber_bpsk(SNR)
        plr_result = FinalChoiceByAttenuationNoise().calculate_packet_loss(ber_result, 100)

        return plr_result

    def checkThreshold(self, attenuationList, task, partitions, considerDistance):
        valid_entries = []

        for i in range(0, len(attenuationList)):
            if attenuationList[i][2] != 0:
                # todo : add transmit power check ! note: removed!
                # TRANSMISSION_LIST = Cnf.VehicleConfig.TRANSMISSION_LIST
                # for j in range(0, len(TRANSMISSION_LIST)):
                #     plr_result = self.calcPlr(attenuationList, attenuationList[i][2], task, partitions, TRANSMISSION_LIST[i])
                #
                #     # print(red_bg(f"Pr:{Pr}, attenuation: {attenuationList[i][2]}, avgNoise:{avgNoise}, SNR:{SNR}, ber_result:{ber_result}, plr_result:{plr_result}"))
                #     if plr_result < FinalChoiceByAttenuationNoise().THERESHOLD:
                #         valid_entries.append((attenuationList[i], plr_result))
                #         break

                plr_result = self.calcPlr(attenuationList, attenuationList[i][2], task, partitions)
                # print(red_bg(f"Pr:{Pr}, attenuation: {attenuationList[i][2]}, avgNoise:{avgNoise}, SNR:{SNR}, ber_result:{ber_result}, plr_result:{plr_result}"))
                if plr_result < FinalChoiceByAttenuationNoise().THERESHOLD:
                    valid_entries.append((attenuationList[i], plr_result))

        if valid_entries:
            if considerDistance:
                print("teeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeestttttttttttttttttttttttttttttttttt")
                return self.findMinValidEntries(valid_entries)


            else:
                best_entry = min(valid_entries, key=lambda x: x[1])
                # print(f"best_entry: {best_entry}")
                return best_entry
        else:
            for i in range(0, len(attenuationList)):
                if attenuationList[i][2] == 0:
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
        return ber

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
            plr_result = self.calcPlr(attenuationList, data[2], task, partitions)
            return data, plr_result
        else:
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
        for i in range(0, len(attenuationList)):
            distances.append(
                (attenuationList[i], self.calcMinDistance(attenuationList[i][0].zone, attenuationList[i][1])))
        min_distance = min(distances, key=lambda x: x[1])
        return min_distance

    def findMinValidEntries(self, valid_entries):
        distances = []
        for i in range(0, len(valid_entries)):
            distances.append(
                (valid_entries[i], self.calcMinDistance(valid_entries[i][0][0].zone, valid_entries[i][0][1])))
            print(red_bg(f"distances : {distances[i][1]}"))

        min_distance = min(distances, key=lambda x: x[1])
        return min_distance[0]


if __name__ == "__main__":
    # Example usage:
    snr_input = float(input("Enter SNR (in dB): "))  # User inputs SNR value
    ber_result = FinalChoiceByAttenuationNoise().calculate_ber_bpsk(snr_input)
    plr_result = FinalChoiceByAttenuationNoise().calculate_packet_loss(ber_result, 1000)

    print(f"SNR = {snr_input} dB → BER = {ber_result:.16f}")
    print(f"PLR = {plr_result:.16f}%")
