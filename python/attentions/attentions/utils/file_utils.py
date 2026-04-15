#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
from functools import reduce
from .exception import ParametersInvalid
from .logs.logging import logger

MAX_PATH_LENGTH = 4096
MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024
MAX_FILENUM_PER_DIR = 1024
MAX_LINENUM_PER_FILE = 10 * 1024 * 1024
SAFEOPEN_FILE_PERMISSION = 0o640
CONFIG_FILE_PERMISSION = 0o640
MODELDATA_FILE_PERMISSION = 0o640
MODELDATA_DIR_PERMISSION = 0o750
BINARY_FILE_PERMISSION = 0o755

FLAG_OS_MAP = {
    'r': os.O_RDONLY, 'r+': os.O_RDWR,
    'w': os.O_CREAT | os.O_TRUNC | os.O_WRONLY,
    'w+': os.O_CREAT | os.O_TRUNC | os.O_RDWR,
    'a': os.O_CREAT | os.O_APPEND | os.O_WRONLY,
    'a+': os.O_CREAT | os.O_APPEND | os.O_RDWR,
    'x': os.O_CREAT | os.O_EXCL,
    "b": getattr(os, "O_BINARY", 0)
}


def safe_open(file_path: str, mode='r', encoding=None, permission_mode=0o640, **kwargs):
    """
    Args:
        file_path (str): 文件路径
        mode (str): 文件打开模式
        encoding (str): 文件编码方式
        permission_mode: 文件权限最大值
        max_path_length (int): 文件路径最大长度
        max_file_size (int): 文件最大大小，单位: 字节, 默认值10MB
        check_link (bool): 是否校验软链接
        kwargs:
    """
    max_path_length = kwargs.get('max_path_length', MAX_PATH_LENGTH)
    max_file_size = kwargs.get('max_file_size', MAX_FILE_SIZE)
    check_link = kwargs.get('check_link', True)

    file_path = standardize_path(file_path, max_path_length, check_link)
    check_file_safety(file_path, max_file_size, permission_mode)

    flags = []
    for item in list(mode):
        if item == "+" and flags:
            flags[-1] = f"{flags[-1]}+"
            continue
        flags.append(item)
    flags = [FLAG_OS_MAP.get(mode, os.O_RDONLY) for mode in flags]
    total_flag = reduce(lambda a, b: a | b, flags)

    return os.fdopen(os.open(file_path, total_flag, SAFEOPEN_FILE_PERMISSION),
                     mode, encoding=encoding)


def standardize_path(path: str, max_path_length=MAX_PATH_LENGTH, check_link=True):
    """
    Check and standardize path.
    Args:
        path (str): 未标准化路径
        max_path_length (int): 文件路径最大长度
        check_link (bool): 是否校验软链接
    Return: 
        path (str): 标准化后的绝对路径
    """
    check_path_is_none(path)
    check_path_length_lt(path, max_path_length)
    if check_link:
        check_path_is_link(path)
    path = os.path.realpath(path)
    return path


def is_path_exists(path: str):
    return os.path.exists(path)


def check_path_is_none(path: str):
    if path is None:
        raise ParametersInvalid("The path should not be None.")


def check_path_is_link(path: str):
    if os.path.islink(os.path.normpath(path)):
        raise ParametersInvalid(f"The path:{os.path.relpath(path)} is a symbolic link file.")


def check_path_length_lt(path: str, max_path_length=MAX_PATH_LENGTH):
    if path.__len__() > max_path_length:
        raise ParametersInvalid(f"The length of path is {path.__len__()}, which exceeds the limit {max_path_length}.")


def check_file_size_lt(path: str, max_file_size=MAX_FILE_SIZE):
    if os.path.getsize(path) > max_file_size:
        raise ParametersInvalid(
            f"The size of file:{os.path.relpath(path)} is {os.path.getsize(path)}, \
                which exceeds the limit {max_file_size}.")


def check_owner(path: str):
    path_stat = os.stat(path)
    path_owner, path_gid = path_stat.st_uid, path_stat.st_gid
    cur_uid = os.geteuid()
    cur_gid = os.getgid()
    if not (cur_uid == 0 or cur_uid == path_owner or path_gid == cur_gid):
        raise ParametersInvalid(f"The current user does not have permission to access the path:{path}. "
                                "Because he is not root or the path owner, "
                                "and not in the same user group with the path owner. "
                                "Please check and make sure to satisfy at least one of the conditions above.")


def check_max_permission(file_path: str, permission_mode=0o640):
    current_permissions = os.stat(file_path).st_mode & 0o777
    required_permissions = permission_mode & 0o777
    for i in range(3):
        cur_perm = (current_permissions >> (i * 3)) & 0o7
        max_perm = (required_permissions >> (i * 3)) & 0o7
        if (cur_perm | max_perm) != max_perm:
            err_msg = f"The permission of {os.path.relpath(file_path)} is higher than {oct(required_permissions)}."
            raise PermissionError(err_msg)


def check_file_safety(file_path: str, max_file_size=MAX_FILE_SIZE, is_check_file_size=True, permission_mode=0o640):
    if not is_path_exists(file_path):
        raise ParametersInvalid(f"The path:{os.path.relpath(file_path)} doesn't exist.")
    if not os.path.isfile(file_path):
        raise ParametersInvalid(f"The input:{os.path.relpath(file_path)} is not a file.")
    if is_check_file_size:
        check_file_size_lt(file_path, max_file_size)
    check_owner(file_path)
    check_max_permission(file_path, permission_mode)


def check_dir_safety(dir_path: str, max_file_num=MAX_FILENUM_PER_DIR, is_check_file_num=True, permission_mode=0o750):
    if not is_path_exists(dir_path):
        raise ParametersInvalid(f"The path:{os.path.relpath(dir_path)} doesn't exist.")
    if not os.path.isdir(dir_path):
        raise ParametersInvalid(f"The path:{os.path.relpath(dir_path)} is not a dir.")
    if is_check_file_num:
        check_file_num_lt(dir_path, max_file_num)
    check_owner(dir_path)
    check_max_permission(dir_path, permission_mode)


def check_file_num_lt(path: str, max_file_num=MAX_FILENUM_PER_DIR):
    filenames = os.listdir(path)
    if len(filenames) > max_file_num:
        raise ParametersInvalid(
            f"The number of files in dir:{os.path.relpath(path)} is {len(filenames)}, \
                which exceeds the limit {max_file_num}."
        )


def safe_listdir(file_path: str, max_file_num=MAX_FILENUM_PER_DIR):
    filenames = os.listdir(file_path)
    if len(filenames) > max_file_num:
        raise ParametersInvalid(
            f"The number of files in dir:{os.path.relpath(file_path)} is {len(filenames)}, \
                which exceeds the limit {max_file_num}."
        )
    return filenames


def safe_readlines(file_obj, max_line_num=MAX_LINENUM_PER_FILE):
    lines = file_obj.readlines()
    if len(lines) > max_line_num:
        raise ParametersInvalid(f"The number of lines in file is {len(lines)}, which exceeds the limit {max_line_num}.")
    return lines


# check every files under dir_path
def check_file_under_dir(dir_path):
    path_check = standardize_path(dir_path)
    check_dir_safety(path_check, permission_mode=MODELDATA_DIR_PERMISSION)
    files = os.listdir(dir_path)
    for file in files:
        file_path = os.path.join(dir_path, file)

        if os.path.isfile(file_path):
            path_check = standardize_path(file_path)
            check_file_safety(path_check)