#
# Copyright (c) 2014-2024 - UPMEM
# Licensed under the MIT License. See the LICENSE file in this directory.
#


def add_dictionaries(dict1, dict2):
    return {
        key: dict1.get(key, 0) + dict2.get(key, 0) for key in set(dict1) | set(dict2)
    }
