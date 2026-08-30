from pulseppg.nets.Base_Nets import Base_NetConfig

from pulseppg.eval.Base_Eval import Base_EvalConfig
from pulseppg.data.Base_Dataset import SSLDataConfig, SupervisedDataConfig

from pulseppg.models.RelCon.RelCon_Model import RelCon_ModelConfig

WAVELENGTHS = ["660", "730", "850", "940"]

allpulseppg_expconfigs = {}

allpulseppg_expconfigs["pulseppg"] = RelCon_ModelConfig(
    withinuser_cands=1,
    encoder_dims=[0],

    motifdist_expconfig_key="motifdist",

    data_config=SSLDataConfig(
        data_folder="none",
        data_normalizer_path = "none", 
        data_clipping = True, 
    ),

    net_config=Base_NetConfig(
        net_folder="ResNet1D",
        net_file="ResNet1D_Net",
        params = {"in_channels":1,
                  "base_filters": 128,
                  "kernel_size": 11, # 15 -> 30 -> 60 -> 120 -> 240 -> 480
                  "stride":2,
                  "groups": 1,
                  "n_block": 12,
                  "finalpool": "max"}
    ),
    epochs = 20, lr=0.0001, batch_size=16, save_epochfreq=1,
    eval_configs = [
        Base_EvalConfig(
            name=f"embeddings_output_ppg_glucose_50hz_{wl}nm",
            model_folder="Regress",
            model_file="linear_probe",
            data_config=SupervisedDataConfig(
                data_folder="pulseppg/data/datasets/Hb_PPG_Dataset/",
                X_annotates=[f"_ppg_50Hz_{wl}nm"],
                y_annotate=f"_glucose_{wl}nm",
            ),
        )
        for wl in WAVELENGTHS
    ]
)
