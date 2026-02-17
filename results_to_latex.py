import glob 
import os 
import json 
import pandas as pd

# 1. Setup Data and Mappings
no_folds = [
    "speech_commands-v0.0.2-5h",
    "dcase2016_task2-hear2021-full",
    "nsynth_pitch-v2.2.3-5h",
    "fsd50k-v1.0-full",
]

# Updated mapping to include WavJEPA and fix WavLM Large to "Mix"
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
    "wavjepa_base": ["WavJEPA", "B", "LS", "960"],
    "wavjepa_as_base": ["WavJEPA", "B", "AS", "4.7k"],
}

data_name_mapping = {
    "beijing_opera-v1.0-hear2021-full" : "BO",
    "dcase2016_task2-hear2021-full" : "DCASE",
    "esc50-v2.0.0-full" : "ESC-50",
    "fsd50k-v1.0-full" : "FSD50K",
    "libricount-v1.0.0-hear2021-full" : "LC",
    "mridangam_stroke-v1.5-full" : "Mri-S",
    "mridangam_tonic-v1.5-full" : "Mri-T",
    "nsynth_pitch-v2.2.3-5h" : "NS",
    "speech_commands-v0.0.2-5h" : "SC-5",
    "tfds_crema_d-1.0.0-full" : "CD",
    "vox_lingua_top10-hear2021-full": "VL"
}

# The target order for the DataFrame columns
column_names = ["Model", "Size", "Data", "Data in hours",
                "DCASE", "FSD50K", "LC", "ESC-50",
                "CD", "VL", "SC-5",
                "NS", "BO", "Mri-S", "Mri-T"]

# The exact target order for the rows (as tuples of Model, Size, Data, Hours)
row_order = [
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
    ("HuBERT", "L", "AS", "4.3k")
]

# 2. Helper Function
def load_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Safely get the dataset name from the path (cross-platform compatible)
    dataset_name = os.path.basename(os.path.dirname(json_path))
    
    if dataset_name in no_folds: 
        return str(round(data["test"]["test_score"] * 100, 1))
    else:
        return str(round(data["aggregated_scores"]["test_score_mean"] * 100, 1)) + " \pm " + str(
            round(data["aggregated_scores"]["test_score_std"] * 100, 1)
        )

# 3. Build the Data
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

            model_path = os.path.join("hear_scores", path)
            if os.path.isdir(model_path):
                for dir_name in os.listdir(model_path):
                    if dir_name in data_name_mapping:
                        json_file = os.path.join(model_path, dir_name, "test.predicted-scores.json")
                        try:
                            score_str = load_data(json_file)
                            column_header = data_name_mapping[dir_name]
                            row_data[column_header] = score_str
                        except FileNotFoundError as e:
                            pass # File missing, will become NaN in pandas
                        except KeyError as e:
                            pass # JSON structured differently

            rows.append(row_data)

# 4. Create Pandas DataFrame
df = pd.DataFrame(rows)

# 5. Arrange Columns
df = df.reindex(columns=column_names)

# 6. Apply Custom Row Sorting
if not df.empty:
    # Create a temporary rank column for sorting based on the row_order list
    def get_sort_rank(row):
        model_tuple = (row['Model'], row['Size'], row['Data'], row['Data in hours'])
        try:
            return row_order.index(model_tuple)
        except ValueError:
            return 999  # Put any unlisted models at the bottom

    df['SortRank'] = df.apply(get_sort_rank, axis=1)
    
    # Sort and remove the temporary column
    df = df.sort_values(by='SortRank').drop(columns=['SortRank']).reset_index(drop=True)

# Replace missing values (NaN) with an empty string or "-" for cleaner viewing
df = df.fillna("-")

print(df)