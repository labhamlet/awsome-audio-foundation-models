# awsome-audio-foundation-models
This repo contains benchmarking code for recent audio foundation models on HEAR and Nat-HEAR datasets.
We have collected 23 recent audio foundation models from the existing literature. Here are all the models we managed to collect, along with open-source weights and implementations.

## Models

| Model    | Venue | Input Feature| Pre-Training Dataset Hours | GPU Hours |  Description | 
| -------- | -------  | ------- | ------- | ------- | ------- 
| **Input Reconstruction** | | | | |
| | | | | |
| GRAM  | ArXiv 2025| Mel Spectrogram| 4.7k | | |
| MWMAE    |   ICLR 2024  | Mel Spectrogram | 5.5k | | |
| SSAM    |   INTERSPEECH 2024  | Mel Spectrogram | 5.5k | | |
| AudioMAE    |  NeurIPS 2022 | Mel Spectrogram | 5.5k | | |
| MSM-MAE    |  NeurIPS 2021 Competition   | Mel Spectrogram| 5.4k | | |
| Dasheng    |  INTERSPEECH 2024    | Mel Spectrogram | 272k | | |
| **Latent Space Reconstruction** | | | | | |
| | | | | |
| BYOL-A | TASLP 2023 | Mel Spectrogram| 5.4k | | |
| M2D    | TASLP 2024    |Mel Spectrogram | 5.4k | | |
| USAD | ArXiv | Mel Spectrogram | 161k | | |
| ATST-Clip | ICASSP 2024 | Mel Spectrogram | 5.4k | | |
| ATST-Frame | ICASSP 2024 | Mel Spectrogram | 5.4k| | |
| WavJEPA | ArXiv 2025 | Raw Waveform | 4.7k | | |
| **Latent Space Prediction** | | | | | |
| | | | | |
| BEATs |  ICML 2023 | Mel Spectrogram | 5.4k| | |
| SSAST |  AAAI 2022 | Mel Spectrogram | 6.2k | | |
| **Speech (Latent Space Reconstruction/Prediction)** | | | | | |
| | | | | |
| WavLM    |  JSTSP 2022 | Raw Waveform | 94k| | |
| HuBERT    | TASLP 2021 | Raw Waveform | 61k | | |
| Wav2Vec2.0  |  NeurIPS 2020 | Raw Waveform | 53.2k | | |
| Whisper   |  ICML 2023 | Mel Spectrogram | 681k | | |
| Data2Vec   | ICML 2022 | Raw Waveform | 960 | | |
| BYOL-S |  NeurIPS Competition HEAR | Mel Spectrogram | 2.1k | | |
| UniSpeech | ICML 2021 | Raw Waveform | 5k | | | |
| **Supervised** | | | | | |
| | | | | |
| PASST    | INTERSPEECH 2022 | Mel Spectrogram | 5.4k | |  |
| Spatial-AST    | ICML 2024  | Mel Spectrogram + IPD | 5.4k | | |

# Results on HEAR Benchmark


# Results on Nat-HEAR Benchmark
