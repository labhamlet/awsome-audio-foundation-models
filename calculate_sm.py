import os 
import json 
import pandas as pd
import numpy as np


def sm(baseline, models):
  sota = np.max(np.concatenate([baseline, models]), axis = 0)
  s = (models - baseline) / ((sota - baseline) + 0.0001)
  s[s < 0] = 0
  return np.round(100 * s.mean(axis = 1),1)


baseline = [
    7.6,
    12.5,
    40.3,
    27.4,
    36.7,
    16.0,
    13.5,
    89.2,
    97.1,
    94.2,
    93.7
]

no_folds = [
    "speech_commands-v0.0.2-5h",
    "dcase2016_task2-hear2021-full",
    "nsynth_pitch-v2.2.3-5h",
    "fsd50k-v1.0-full",
]

spectrogram_models = [
    "atst_clip",
    "dasheng",
    "atst_frame",
    "beats",
    "ssast_400_patch",
    "mwmae_base_200_4x16_384d-8h-4l",
    "MAE",
]

model_name_mapping = {
    "data2vec_base": ["Data2Vec", "B", "LS", "960"],
    "data2vec_large": ["Data2Vec", "L", "LS", "960"],
    "wav2vec2_base": ["Wav2Vec2.0", "B", "LS", "960"],
    "wav2vec2_large": ["Wav2Vec2.0", "L", "LV", "60k"],
    "hubert_base": ["HuBERT", "B", "LS", "960"],
    "hubert_large": ["HuBERT", "L", "LV", "60k"],
    "wavlm_base": ["WavLM", "B", "LS", "960"],
    "wavlm_large": ["WavLM", "L", "Mix", "94k"],
    "wav2vec2_as_base": ["Wav2Vec2.0", "B", "AS", "4.3k"],
    "wav2vec2_as_large": ["Wav2Vec2.0", "L", "AS", "4.3k"],
    "hubert_as_base": ["HuBERT", "B", "AS", "4.3k"],
    "hubert_as_large": ["HuBERT", "L", "AS", "4.3k"],
    "WavJEPA_ls": ["WavJEPA", "B", "LS", "960"], 
    "WavJEPA_as": ["WavJEPA", "B", "AS", "4.7k"],
    "atst_clip": ["ATST-Clip", "B", "AS", "5.3k"],
    "dasheng" : ["Dasheng", "B", "Mix", "272k"],
    "atst_frame" : ["ATST-Frame", "B", "AS", "5.3k"],
    "beats" : ["BEATs", "B", "AS", "5.3k"],
    "MAE" : ["AudioMAE", "B", "AS", "5.3k"],
    "ssast_400_patch": ["SSAST", "B", "AS", "5.3k"],
    "mwmae_base_200_4x16_384d-8h-4l": ["MWMAE", "B", "AS", "5.3k"]

}

data_name_mapping = {
    "dcase2016_task2-hear2021-full" : "DCASE",
    "fsd50k-v1.0-full" : "FSD50K",
    "libricount-v1.0.0-hear2021-full" : "LC",
    "esc50-v2.0.0-full" : "ESC-50",
    "tfds_crema_d-1.0.0-full" : "CD",
    "vox_lingua_top10-hear2021-full": "VL",
    "speech_commands-v0.0.2-5h" : "SC-5",
    "nsynth_pitch-v2.2.3-5h" : "NS",
    "beijing_opera-v1.0-hear2021-full" : "BO",
    "mridangam_stroke-v1.5-full" : "Mri-S",
    "mridangam_tonic-v1.5-full" : "Mri-T",
}

column_names = ["Model", "Size", "Data", "Data in hours",
                "DCASE", "FSD50K", "LC", "ESC-50",
                "CD", "VL", "SC-5",
                "NS", "BO", "Mri-S", "Mri-T"]

# Define the exact sort order requested
sort_order = [
    ("Wav2Vec2.0", "B", "LS", "960"),
    ("HuBERT", "B", "LS", "960"),
    ("WavLM", "B", "LS", "960"),
    ("Data2Vec", "B", "LS", "960"),
    ("WavJEPA", "B", "LS", "960"),
    ("Wav2Vec2.0", "L", "LV", "60k"),
    ("HuBERT", "L", "LV", "60k"),
    ("WavLM", "L", "Mix", "94k"),
    ("Data2Vec", "L", "LS", "960"),
    ("Wav2Vec2.0", "B", "AS", "4.3k"),
    ("HuBERT", "B", "AS", "4.3k"),
    ("WavJEPA", "B", "AS", "4.7k"),
    ("Wav2Vec2.0", "L", "AS", "4.3k"),
    ("HuBERT", "L", "AS", "4.3k"),
    ("AudioMAE", "B", "AS", "5.3k"),
    ("SSAST", "B", "AS", "5.3k"),
    ("BEATs", "B", "AS", "5.3k"),
    ("MWMAE", "B", "AS", "5.3k"),
    ("Dasheng", "B", "Mix", "272k"),
    ("ATST-Clip", "B", "AS", "5.3k"),
    ("ATST-Frame", "B", "AS", "5.3k"),  

]

def load_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    if json_path.split("/")[-2] in no_folds: 
        return str(round(data["test"]["test_score"] * 100, 1))
    else:
        return str(round(data["aggregated_scores"]["test_score_mean"] * 100, 1)) + r" \pm " + str(
            round(data["aggregated_scores"]["test_score_std"] * 100, 1)
        )
    
rows = []
if os.path.exists("hear_scores"):
    for path in os.listdir("hear_scores"):
        model_key = path.split(".")[-1]
        if model_key in model_name_mapping:
            model_info = model_name_mapping[model_key]
            row_data = {
                "Model": model_info[0],
                "Size": model_info[1],
                "Data": model_info[2],
                "Data in hours": model_info[3]
            }
            
            # Helper to calculate row average
            scores_for_avg = []
            scores_for_sm = []

            model_path = os.path.join("hear_scores", path)
            for dir_name in os.listdir(model_path):
                if dir_name in data_name_mapping:
                    json_file = os.path.join(model_path, dir_name, "test.predicted-scores.json")
                    try:
                        score_str = load_data(json_file)
                        col = data_name_mapping[dir_name]
                        row_data[col] = score_str
                        # Extract numeric value for average calculation
                        # Takes "60.4" from "60.4 \pm 1.2"
                        val = float(score_str.split(" ")[0])
                        scores_for_avg.append(val)
                    except Exception:
                        #Some models do not have DCASE, append 0 for s(m)
                        pass
            
            if scores_for_avg:
                row_data["avg"] = f"{np.mean(scores_for_avg):.1f}"
            else:
                row_data["avg"] = "-"
            rows.append(row_data)

df = pd.DataFrame(rows)

# --- 3. SORTING ---

# Create a temporary column for sorting index
def get_sort_index(row):
    key = (row["Model"], row["Size"], row["Data"], row["Data in hours"])
    return sort_order.index(key)

df["_sort_id"] = df.apply(get_sort_index, axis=1)
df = df.sort_values("_sort_id").drop("_sort_id", axis=1)

score_columns = ["Model", "DCASE", "FSD50K", "LC", "ESC-50", "CD", "VL", "SC-5", "NS", "BO", "Mri-S", "Mri-T"]
df_s = df.reindex(columns=score_columns).to_numpy()
df_s = df_s[:-7, 1:]
scores = np.zeros((df_s.shape[0], df_s.shape[1]))
for i, rows in enumerate(df_s):
    for j, item in enumerate(rows): 
        scores[i,j] = float(item.split(" ")[0])

sms = np.concatenate([sm([baseline], scores), ["nan","nan","nan","nan","nan","nan","nan"]])
df["s(m)"] = sms

# --- 4. LATEX GENERATION ---
# Extract the display names for the spectrogram models from the mapping dictionary
score_columns = ["DCASE", "FSD50K", "LC", "ESC-50", "CD", "VL", "SC-5", "NS", "BO", "Mri-S", "Mri-T", "avg", "s(m)"]

spectro_display_names = [model_name_mapping[k][0] for k in spectrogram_models]

# Iterate over dataframe to print rows
for idx, row in df.iterrows():
    
    # Check if current model is one of the spectrogram models
    is_spectro = row["Model"] in spectro_display_names
    c_tag = r"\color{gray} " if is_spectro else ""
    
    line_items = []
    
    # Metadata columns
    line_items.extend([
        f"{c_tag}{row['Model']}", 
        f"{c_tag}{row['Size']}", 
        f"{c_tag}{row['Data']}", 
        f"{c_tag}{row['Data in hours']}"
    ])
    
    # Score columns
    for col in score_columns:
        val = str(row[col])
        if r"\pm" in val:
            # Wrap in math mode if it has standard deviation
            line_items.append(f"{c_tag}$ {val} $")
        else:
            line_items.append(f"{c_tag}{val}")
            
    # Join with separator
    latex_row = " & ".join(line_items) + r" \\"
    
    print(latex_row)