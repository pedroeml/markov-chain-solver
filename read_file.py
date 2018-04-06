import json


def load_json_file(file_path):
    """

    :param file_path:
    :return:
    :rtype: dict
    """
    with open(file_path, 'r') as f:
        json_file = json.load(f)

    return json_file
