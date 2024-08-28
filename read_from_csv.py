import csv
import os
import global_methods as globalMethod


def getDictionaryFromCSV(csv_file_name):
    try:
        if not os.path.exists(csv_file_name):
            print(f"File not found: {csv_file_name}")
        elif os.path.getsize(csv_file_name) == 0:
            print(f"File is empty: {csv_file_name}")
        else:
            _list = []
            with open(csv_file_name, mode='r') as csvfile:
                csv_file = csv.DictReader(csvfile)
                for line in csv_file:
                    lt_knee = {"lt_knee": globalMethod.getXYZ(line, 'lt_knee')}
                    lt_hip = {"lt_hip": globalMethod.getXYZ(line, 'lt_hip')}
                    lt_ank = {"lt_ank": globalMethod.getXYZ(line, 'lt_ank')}
                    rt_hip = {"rt_hip": globalMethod.getXYZ(line, 'rt_hip')}
                    rt_knee = {"rt_knee": globalMethod.getXYZ(line, 'rt_knee')}
                    rt_ank = {"rt_ank": globalMethod.getXYZ(line, 'rt_ank')}
                    lt_plv = {"lt_plv": globalMethod.getXYZ(line, 'lt_plv')}
                    rt_plv = {"rt_plv": globalMethod.getXYZ(line, 'rt_plv')}
                    task_seq = {"task": int(line['task'])}
                    _dic = {**task_seq, **lt_hip, **lt_knee, **lt_ank, **rt_hip, **rt_knee, **lt_plv, **rt_plv,
                            **rt_ank}
                    _list.append(_dic)
                return _list
    except FileNotFoundError:
        print(f"File not found: {csv_file_name}")
    except Exception as e:
        print(f"An error occurred: {e}")


def get_final_list(normalized_data):
    dictionary_from_scv_data = getDictionaryFromCSV(normalized_data)
    list_with_start_and_end_points = []
    for item in dictionary_from_scv_data:
        list_with_start_and_end_points.append(globalMethod.get_list_of_lines(item))

    return list_with_start_and_end_points
