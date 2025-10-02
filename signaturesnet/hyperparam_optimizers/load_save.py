import os
import json
import pandas


def load(data_file):
    """Load the given data file and returns a dictionary of the values

    Args:
        data_file (str): Path to a file containing data in one of the known formats (json, csv)

    Returns:
        A dictionary with the loaded data

    """
    _, ext = os.path.splitext(data_file)
    try:
        return known_file_types[ext]['load'](data_file)
    except KeyError:
        raise Exception('Error loading file: type ' + str(ext) + ' is not supported')


def save(data_file, dictionary):
    """Save the given dictionary in the given file. Format is determined by data_file extension

    Args:
        data_file (str): Path to a file in which to save the data. Extension is used to determine
        the format, therefore the path must contain an extension.
        dictionary (dict): Dictionary with the data

    """
    _, ext = os.path.splitext(data_file)
    try:
        known_file_types[ext]['save'](data_file, dictionary)
    except KeyError:
        raise Exception('Error loading file: type ' + str(ext) + ' is not supported')


def _load_json(data_file):
    with open(data_file, 'r') as file:
        return json.load(file)


def _save_json(data_file, dictionary):
    with open(data_file, 'w') as file:
        json.dump(dictionary, file)


def _load_csv(data_file):
    data_frame = pandas.read_csv(data_file, header=0)
    res = {}
    for k in data_frame:
        res[k] = data_frame[k].tolist()
    return res


def _save_csv(data_file, dictionary):
    data_frame = pandas.DataFrame.from_dict(data=dictionary, orient='columns')
    data_frame.to_csv(data_file, header=True, index=False)


known_file_types = {
    '.json':
        {
            'load': _load_json,
            'save': _save_json
        },
    '.csv':
        {
            'load': _load_csv,
            'save': _save_csv
        }
}

