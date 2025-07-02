import os
import json
import pandas as pd

def convert_dstc_json_to_csv(json_dir, output_csv):
    data = []
    for split in ['train', 'dev', 'test']:
        split_dir = os.path.join(json_dir, split)
        if not os.path.exists(split_dir):
            continue
        for filename in os.listdir(split_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(split_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    dialogues = json.load(f)
                    for dialogue in dialogues:
                        dialogue_id = dialogue.get('dialogue_id', '')
                        for turn in dialogue.get('turns', []):
                            speaker = turn.get('speaker', '')
                            utterance = turn.get('utterance', '')
                            data.append({
                                'ID': dialogue_id,
                                'Input': speaker,
                                'Text': utterance
                            })
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding='utf-8')

# Example usage:
convert_dstc_json_to_csv('BrainEmbeddings/dstc8', 'dstc8_full_conversations.csv')
