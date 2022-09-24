
import os
import time
from typing import List


def list_of_dicts_2_list_of_strings(lst: List[dict]) -> List[str]:
    """Make list of strings from list of dicts with ordered keys.
    Assuming that keys are equal in all dicts. Otherwise, the error will be
    occurred or some keys might be lost."""
    columns = sorted(lst[0].keys())
    values = [','.join([str(dct[k]) for k in columns]) + '\n' for dct in lst]
    header = [','.join(columns) + '\n']
    return header + values


class ContainerCSV:
    def __init__(self, out_dir_path: str):
        current_time = time.strftime('%Y-%m-%d_%H-%M-%S')
        self.filename = f'output_{current_time}'
        self.out_dir_path = out_dir_path
        self.storage = []  # List of dicts.
                           # Assuming that keys are equal in all dicts.

    def save(self) -> str:
        lines = list_of_dicts_2_list_of_strings(self.storage)
        filepath = os.path.join(self.out_dir_path, self.filename) + '.csv'
        with open(filepath, 'w') as output_file:
            output_file.writelines(lines)
        print(f'Data has been saved into .csv file: {filepath}')
        return filepath

    def write_line(self, line: dict):
        if line is not None:
            self.storage.append(line)


if __name__ == '__main__':
    csv = ContainerCSV('../csv_output')

    import random
    test_storage = [{k: random.randint(1, 100) for k in 'abcde'}
                    for _ in range(10)]
    for dct in test_storage:
        print([v for v in dct.values()])
        csv.write_line(dct)

    csv.storage = test_storage
    csv.save()
