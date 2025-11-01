class NoiseConfigGeneralAttribute:
    Rain_options = [
        "NoiseConfig.Rain0()",
        "NoiseConfig.Rain13()",
        "NoiseConfig.Rain23()",
        "NoiseConfig.Rain50()",
        "NoiseConfig.Rain100()",
        "NoiseConfig.Rain150()",
        "NoiseConfig.Rain200()",
    ]

    Rain_class_to_unit = {
        "Rain0": 0, "Rain13": 1, "Rain23": 2, "Rain50": 3,
        "Rain100": 4, "Rain150": 5, "Rain200": 6
    }

    Traffic_options = [
        "NoiseConfig.BlackTraffic()",
        "NoiseConfig.RedTraffic()",
        "NoiseConfig.OrangeTraffic()",
        "NoiseConfig.YellowTraffic()",
        "NoiseConfig.GreenTraffic()",
    ]

    Urban_options = [
        "NoiseConfig.Rural()",
        "NoiseConfig.GoodUrban()",
        "NoiseConfig.MediumUrban()",
        "NoiseConfig.BadUrban()",
    ]
