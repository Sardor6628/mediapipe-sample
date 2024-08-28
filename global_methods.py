def getXKey(key):
    return key + "_sagittal"


def getYKey(key):
    return key + "_frontal"


def getZKey(key):
    return key + '_transe'


def getXYZ(line, key):
    return [float(line[getXKey(key)]),
            float(line[getYKey(key)]),
            float(line[getZKey(key)])]


def get_line(start_points, end_points):
    return {"x": [start_points[0], end_points[0]], "y": [start_points[1], end_points[1]],
            "z": [start_points[2], end_points[2]]}


def get_list_of_lines(_dict):
    _list = [
        get_line(_dict['lt_plv'], _dict['rt_plv']),
        get_line(_dict['lt_hip'], _dict['rt_hip']),
        get_line(_dict['rt_hip'], _dict['rt_knee']),
        get_line(_dict['rt_knee'], _dict['rt_ank']),
        get_line(_dict['lt_hip'], _dict['lt_knee']),
        get_line(_dict['lt_knee'], _dict['lt_ank']),
        get_line(_dict['rt_plv'], _dict['rt_hip']),
        get_line(_dict['lt_plv'], _dict['lt_hip'])
    ]
    return _list
