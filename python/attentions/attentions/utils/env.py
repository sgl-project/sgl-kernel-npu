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
import re
from dataclasses import dataclass


VALID_LOG_LEVELS = ["critical", "error", "warn", "info", "debug", "null"]
VALID_BOOLEAN = ["true", "false", "1", "0"]
VALID_CYCLE = ["daily", "weekly", "monthly", "yearly"]
MAX_STRING_LENGTH = 256
COMPONENT_NAME = "sd"
ALL_COMPONENT_NAME = "*"
MAX_FILE_SIZE = 500
MAX_ROTATE_DAY = 180
MAX_FILE_NUM = 64


def parser_env_to_dict(mindie_log_level, valid_keys=None):
    log_level = {}
    # get log_level
    modules = mindie_log_level.split(';')
    for module in modules:
        module = module.strip()
        if ":" in module:
            module_name, level = module.split(':')[:2] # 2: only retrieve the first two
            level = level.strip().lower()
            if valid_keys is None or level in valid_keys:  
                log_level[module_name.strip()] = level
        module = module.lower() # A~Z to a~z
        if valid_keys is None or module in valid_keys:
            log_level[ALL_COMPONENT_NAME] = module
    return log_level


def check_string_valid(env_str: str) -> bool:
    invalid_chars = {
        '\f', '\r', '\b', '\t', '\v', '\n',
        '\u000A', '\u000D', '\u000C', '\u000B',
        '\u0008', '\u007F', '\u0009'
    }
    for char in invalid_chars:
        if char in env_str:
            return False
    return True


@dataclass
class EnvVar:
    """
    Environment variable.
    """
    # The log level, supports input of [critical, error, warn, info, debug, null]
    mindie_log_level: str = os.getenv("MINDIE_LOG_LEVEL", "info")
    # Whether to print logs, the default is `true`, supports input of [true, false, 1, 0].
    mindie_log_stdout: str = os.getenv("MINDIE_LOG_TO_STDOUT", "true")
    # Whether to write logs, the default is `true`, supports input of [true, false, 1, 0].
    mindie_log_to_file: str = os.getenv("MINDIE_LOG_TO_FILE", "true")
    # The path to write logs, the default is `~/mindie/log`.
    mindie_log_path: str = os.getenv("MINDIE_LOG_PATH", "~/mindie/log")
    
    mindie_log_verbose: str = os.getenv("MINDIE_LOG_VERBOSE", "true")

    mindie_log_rotate: str = os.getenv("MINDIE_LOG_ROTATE", "-s 30 -fs 20 -r 10")

    disable_log: bool = False

    component_log_level: str = "info"

    component_log_stdout: str = "true"

    component_log_to_file: str = "true"

    component_log_path: str = "~/mindie/log/"

    component_log_verbose: str = "true"

    rotate_cycle: str = "daily"

    rotate_cycle_num: int = 30

    rotate_max_file_size: int = 20 # MB

    rotate_max_file_num: int = 10

    def __post_init__(self):
        self._check()

        log_level = parser_env_to_dict(self.mindie_log_level, VALID_LOG_LEVELS)
        if log_level.get(ALL_COMPONENT_NAME, "") == "null" or log_level.get(COMPONENT_NAME, "") == "null":
            self.disable_log = True
            return
        self.component_log_level = log_level.get(COMPONENT_NAME,
            log_level.get(ALL_COMPONENT_NAME, self.component_log_level))
        
        log_stdout = parser_env_to_dict(self.mindie_log_stdout, VALID_BOOLEAN)
        self.component_log_stdout = log_stdout.get(COMPONENT_NAME,
            log_stdout.get(ALL_COMPONENT_NAME, self.component_log_stdout))

        log_to_file = parser_env_to_dict(self.mindie_log_to_file, VALID_BOOLEAN)
        self.component_log_to_file = log_to_file.get(COMPONENT_NAME,
            log_to_file.get(ALL_COMPONENT_NAME, self.component_log_to_file))

        log_path = parser_env_to_dict(self.mindie_log_path)
        self.component_log_path = log_path.get(COMPONENT_NAME,
            log_path.get(ALL_COMPONENT_NAME, self.component_log_path))

        log_verbose = parser_env_to_dict(self.mindie_log_verbose, VALID_BOOLEAN)
        self.component_log_verbose = log_verbose.get(COMPONENT_NAME,
            log_verbose.get(ALL_COMPONENT_NAME, self.component_log_verbose))

        log_rotate = parser_env_to_dict(self.mindie_log_rotate)
        component_log_rotate = log_rotate.get(COMPONENT_NAME,
            log_rotate.get(ALL_COMPONENT_NAME, None))
        self._get_rotate_parameter(component_log_rotate)

    def _check(self):
        if len(self.mindie_log_level) > MAX_STRING_LENGTH:
            raise ValueError(f"The length of the environment variable MINDIE_LOG_LEVEL " \
                f"[{len(self.mindie_log_level)}] is > {MAX_STRING_LENGTH}.")
        if not check_string_valid(self.mindie_log_level):
            raise ValueError(f"The environment variable MINDIE_LOG_LEVEL is invalid!")
        
        if len(self.mindie_log_stdout) > MAX_STRING_LENGTH:
            raise ValueError(f"The length of the environment variable MINDIE_LOG_TO_STDOUT" \
                f"[{len(self.mindie_log_stdout)}] is > {MAX_STRING_LENGTH}.")
        if not check_string_valid(self.mindie_log_stdout):
            raise ValueError(f"The environment variable MINDIE_LOG_TO_STDOUT is invalid!")
        
        if len(self.mindie_log_to_file) > MAX_STRING_LENGTH:
            raise ValueError(f"The length of the environment variable MINDIE_LOG_TO_FILE" \
                f"[{len(self.mindie_log_to_file)}] is > {MAX_STRING_LENGTH}.")
        if not check_string_valid(self.mindie_log_to_file):
            raise ValueError(f"The environment variable MINDIE_LOG_TO_FILE is invalid!")
        
        if len(self.mindie_log_path) > MAX_STRING_LENGTH:
            raise ValueError(f"The length of the environment variable MINDIE_LOG_PATH" \
                f"[{len(self.mindie_log_path)}] is > {MAX_STRING_LENGTH}.")
        if not check_string_valid(self.mindie_log_path):
            raise ValueError(f"The environment variable MINDIE_LOG_PATH is invalid!")

        if len(self.mindie_log_verbose) > MAX_STRING_LENGTH:
            raise ValueError(f"The length of the environment variable MINDIE_LOG_VERBOSE" \
                f"[{len(self.mindie_log_verbose)}] is > {MAX_STRING_LENGTH}.")
        if not check_string_valid(self.mindie_log_verbose):
            raise ValueError(f"The environment variable MINDIE_LOG_VERBOSE is invalid!")
        
        if len(self.mindie_log_rotate) > MAX_STRING_LENGTH:
            raise ValueError(f"The length of the environment variable MINDIE_LOG_ROTATE" \
                f"[{len(self.mindie_log_rotate)}] is > {MAX_STRING_LENGTH}.")
        if not check_string_valid(self.mindie_log_rotate):
            raise ValueError(f"The environment variable MINDIE_LOG_ROTATE is invalid!")

    def _get_rotate_parameter(self, component_log_rotate):
        s_match = re.search(r'-s (\d+)', component_log_rotate) # match '-s 10'
        if s_match is not None:
            self.rotate_cycle_num = int(s_match.group(1)) # num defaults to daily
            if self.rotate_cycle_num < 1 or self.rotate_cycle_num > MAX_ROTATE_DAY:
                raise ValueError(f"The number of days for log rotation must be in range [1, {MAX_ROTATE_DAY}], "
                    f"but got {self.rotate_cycle_num}.")
        
        s_match = re.search(r'-s ([a-z]+)', component_log_rotate) # match '-s daily'
        if s_match is not None and s_match.group(1) in VALID_CYCLE:
            self.rotate_cycle = s_match.group(1)
            self.rotate_cycle_num = 1

        fs_match = re.search(r'-fs (\d+)', component_log_rotate) # match '-fs 1000'
        if fs_match is not None:
            self.rotate_max_file_size = int(fs_match.group(1))
            if self.rotate_max_file_size < 1 or self.rotate_max_file_size > MAX_FILE_SIZE:
                raise ValueError(f"The size of the log rotation file must be in range [1, {MAX_FILE_SIZE}]MB, "
                    f"but got {self.rotate_max_file_size}MB.")
        
        fc_match = re.search(r'-r (\d+)', component_log_rotate) # match '-r 1000'
        if fc_match is not None:
            self.rotate_max_file_num = int(fc_match.group(1))
            if self.rotate_max_file_num < 1 or self.rotate_max_file_num > MAX_FILE_NUM:
                raise ValueError(f"The number of the log rotation file must be in range [1, {MAX_FILE_NUM}], "
                    f"but got {self.rotate_max_file_num}.")


ENV = EnvVar()
