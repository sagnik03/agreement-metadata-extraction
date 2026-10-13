import os

import pandas as pd

from metadata_extractor import MetadataExtractor


FIELD_ORDER = [
    'Agreement Value',
    'Agreement Start Date',
    'Agreement End Date',
    'Renewal Notice (Days)',
    'Party One',
    'Party Two',
]


def main():
    extractor = MetadataExtractor()
    test_df = pd.read_csv('data/test.csv')
    rows = []

    for _, row in test_df.iterrows():
        file_name = str(row['File Name'])
        folder = 'data/test'
        path = os.path.join(folder, file_name)

        if not os.path.exists(path):
            for candidate in os.listdir(folder):
                if candidate.startswith(file_name):
                    path = os.path.join(folder, candidate)
                    break

        if not os.path.exists(path):
            raise FileNotFoundError(f'Could not locate test file: {file_name}')

        prediction = extractor.extract_file(path)
        result = {'File Name': file_name}
        for field in FIELD_ORDER:
            result[field] = prediction.get(field, '')
        rows.append(result)

    output_path = 'data/test_predictions.csv'
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f'Wrote test predictions to {output_path}')


if __name__ == '__main__':
    main()
