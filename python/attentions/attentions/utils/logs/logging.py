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
import sys
import io
from datetime import datetime, timedelta
import re
import logging
from logging.handlers import BaseRotatingHandler
import time

from ..env import ENV


POSITIVE_BOOLEAN = ['true', '1']
MINDIE_DEFAULTS_LOG_PATH = '~/mindie/log/'
MAX_PATH_LEN = 4096
BACKUP_OWNER_SHIP = 0o440
FILE_OWNER_SHIP = 0o640
PATH_OWNER_SHIP = 0o750
MB = 1024 * 1024
MAX_LOG_STRING_LEN = 256

def get_pid():
    return os.getpid()


def get_uid():
    return os.getuid()


def check_owner_permission(file_path, max_mode) -> bool:
    # check owner
    file_owner = os.stat(file_path).st_uid
    cur_owner = get_uid()
    if file_owner != cur_owner:
        logging.warning("File doesn't belong to current user.")
        return False

    # check permission
    file_mode = os.stat(file_path).st_mode & 0o777 # use 777 as mask to get 3-digit octal number
    file_mode_bin = bin(file_mode)[2:].zfill(9) # transeform into 9-bit binary number
    max_mode_bin = bin(max_mode)[2:].zfill(9) # transeform into 9-bit binary number
    for i in range(9): # 9 means 9-bit binary number, checking every bit
        if file_mode_bin[i] > max_mode_bin[i]: # 2 means the head of binary number '0b'
            logging.warning("The permission of file is higher than %s.", oct(max_mode))
            return False

    return True


def check_path(file_path, checking_conf=False):
    # check path length
    if not file_path or len(file_path) >= MAX_PATH_LEN:
        return False

    # check if the path is symbolic link
    trimmed_path = file_path.rstrip("/")
    if os.path.islink(trimmed_path):
        logging.warning("File path is a soft link.")
        return False

    if checking_conf:
        return check_owner_permission(trimmed_path, PATH_OWNER_SHIP)

    return True


class MindIELogFileHandler(BaseRotatingHandler):
    """
    Adapt from logging's TimedRotatingHandler and RotationFileHandler to combine both of their features.
    Beside, add more detail about controlling log files' owner ships and rotation.
    """
    def __init__(self, real_log_path, max_file_num, max_file_size, rotate_cycle_num, rotate_cycle):
        encoding = io.text_encoding(None)
        now_time_str = time.strftime("_%Y%m%d%H%M%S", time.localtime())
        init_log_file = "mindie-sd_" + str(get_pid()) + now_time_str + ".log"
        init_log_path = os.path.realpath(os.path.join(real_log_path, init_log_file))
        
        super().__init__(init_log_path, mode='a', encoding=encoding, delay=True, errors=None)
        self._real_log_path = real_log_path
        self._cur_log_file = init_log_path
        self._max_file_size = max_file_size
        self._max_file_num = max_file_num
        self._rotate_cycle_num = rotate_cycle_num
        self._rotate_cycle = rotate_cycle
        self._next_rollover = self._get_rollover_timepoint() # use time() since it is easier to compute

        # Be aware that, at this point, _cur_log_file is not created yet since the use of delay mode.
        log_path_files = os.listdir(real_log_path)
        log_files = []
        for it in log_path_files:
            real_file_path = os.path.realpath(os.path.join(real_log_path, it))
            time_str = self._get_time_str(it)
            if time_str and real_file_path.startswith(real_log_path) and os.path.exists(real_file_path):
                log_files.append((real_file_path, time_str))
        self._history_files = sorted(log_files, key=lambda x : x[1])
        # Deal with history file number by deleting oldest one
        self._delete_file_by_number()
        self._delete_file_by_time()

    def emit(self, record):
        try:
            if self.should_rollover(record):
                self.do_rollover()
            logging.FileHandler.emit(self, record)
        except Exception:
            self.handleError(record)

    def close(self):
        super().close()
        if os.path.exists(self._cur_log_file):
            os.chmod(self._cur_log_file, BACKUP_OWNER_SHIP)

    def should_rollover(self, record):
        """
        This method will be called when a LogRecord is emitted.
        """
        # check if current log file exist and is valid
        if not os.path.exists(self._cur_log_file):
            return False
        if not os.path.isfile(self._cur_log_file):
            return False

        # check if current log file exceed max file size
        if self.stream is None:
            self.stream = self._open()
        if self._max_file_size > 0:
            msg = "%s\n" % self.format(record)
            self.stream.seek(0, 2)
            if self.stream.tell() + len(msg) >= self._max_file_size:
                return True
        
        # check if the timestamp of the current log exceeds next rollover time
        cur_time = int(time.time())
        if cur_time >= self._next_rollover:
            return True

        return False
    
    def do_rollover(self):
        """
        rotate log file according to file size and time interval
        """
        if self.stream:
            # when doing rollover, close the file object
            self.stream.close()
            self.stream = None

        # add current log file to history
        cur_log_name = os.path.basename(self._cur_log_file)
        cur_time_str = self._get_time_str(cur_log_name)
        self._history_files.append((self._cur_log_file, cur_time_str))

        # delete oldest file by file number constaint
        self._delete_file_by_number()

        # delete oldest file by time
        self._delete_file_by_time()

        self.rotate_file()
        if not self.delay:
            self.stream = self._open()

    def rotate_file(self):
        # modify backup file's owner ship
        os.chmod(self._cur_log_file, BACKUP_OWNER_SHIP)

        # rotate current log file path
        log_file_name = self._get_log_name()
        self._cur_log_file = os.path.join(self._real_log_path, log_file_name)

        # refresh next rollover time point, may have a very slight time difference,
        # but it won't matter since log files are built at least every second.
        self._next_rollover = self._get_rollover_timepoint()
    
    def _get_log_name(self):
        now_time_str = time.strftime("_%Y%m%d%H%M%S", time.localtime())
        log_file_name = "mindie-sd_" + str(get_pid()) + now_time_str + ".log"
        return log_file_name
    
    def _get_time_str(self, file_name):
        log_time = None
        reg = re.compile(R"mindie-sd_(\d+)_(\d{4}\d{2}\d{2}\d{6}).log")
        match = re.match(reg, file_name)
        if match and len(match.groups()) > 1:
            log_pid = match.group(1)
            if log_pid == get_pid():
                log_time = match.group(2)
        return log_time

    def _get_rollover_timepoint(self):
        # rollover timepoint is everyday's midnight, no log file will have log that cross two days.
        now = time.time()
        tomorrow = datetime.fromtimestamp(now) + timedelta(days=1)
        tomorrow_midnight = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, 0) # midnight is 00:00
        rollover_timepoint = int(time.mktime(tomorrow_midnight.timetuple()))
        return rollover_timepoint

    def _delete_file_by_number(self):
        if self._max_file_num > 0:
            while len(self._history_files) >= self._max_file_num:
                self._remove_oldest_log()

    def _delete_file_by_time(self):
        date_format = "%Y%m%d%H%M%S"
        cur_date = datetime.fromtimestamp(time.mktime(time.localtime()))
        while self._history_files:
            oldest_log = self._history_files[0]
            log_time = time.strptime(oldest_log[1], date_format)
            log_date = datetime.fromtimestamp(time.mktime(log_time))
            
            if self._check_time_rotate(log_date, cur_date):
                break
            self._remove_oldest_log()
    
    def _check_daily(self, log_date, cur_date):
        return cur_date - log_date < timedelta(days=self._rotate_cycle_num)

    def _check_weekly(self, log_date, cur_date):
        return cur_date - log_date < timedelta(weeks=self._rotate_cycle_num)

    def _check_month(self, log_date, cur_date):
        if cur_date.year > log_date.year:
            return False
        # 20240306 vs 20240130 rotate_cycle_num=1
        if cur_date.month - log_date.month > self._rotate_cycle_num:
            return False
        # 20240306 vs 20240203 rotate_cycle_num=1, rotate one month and cur_day >= log_day
        if cur_date.month - log_date.month == self._rotate_cycle_num and cur_date.day >= log_date.day:
            return False
        return True

    def _check_year(self, log_date, cur_date):
        # 20240306 vs 20221230 rotate_cycle_num=1
        if cur_date.year - log_date.year > self._rotate_cycle_num:
            return False
        
        if cur_date.year - log_date.year == self._rotate_cycle_num:
            # 20240306 vs 20230208 rotate_cycle_num=1
            if cur_date.month > log_date.month:
                return False
            if cur_date.month == log_date.month:
                # 20240306 vs 20230301
                if cur_date.day >= log_date.day:
                    return False
        return True

    def _check_time_rotate(self, log_date, cur_date):
        match self._rotate_cycle:
            case "daily":
                return self._check_daily(log_date, cur_date)
            case "weekly":
                return self._check_weekly(log_date, cur_date)
            case "monthly":
                return self._check_month(log_date, cur_date)
            case "yearly":
                return self._check_year(log_date, cur_date)
            case _:
                raise ValueError(f"Unknown rotate cycle: {self._rotate_cycle}")

    def _remove_oldest_log(self):
        oldest_log = self._history_files.pop(0)
        if os.path.exists(oldest_log[0]):
            os.remove(oldest_log[0])

    def _open(self):
        create_flags = os.O_RDWR | os.O_CREAT
        open_func = os.fdopen(os.open(self._cur_log_file, create_flags, FILE_OWNER_SHIP),
                              self.mode, encoding=self.encoding, errors=self.errors)
        return open_func


def str_to_loglevel(level_str):
    match level_str.upper():
        case "DEBUG":
            return logging.DEBUG
        case "INFO":
            return logging.INFO
        case "WARN":
            return logging.WARNING
        case "ERROR":
            return logging.ERROR
        case "CRITICAL":
            return logging.CRITICAL
        case _:
            raise ValueError(f"Unknown log level: {level_str}")


class LoggerFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created).astimezone()
        formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + " " + dt.strftime('%z %Z')
        return f"{formatted_time[:23]}{formatted_time[24:27]}:{formatted_time[27:]}" # change 0800 to 08:00

    def format(self, record):
        original = logging.Formatter.format(self, record)
        return self._filter(original)

    def _filter(self, message):
        if message is not None:
            if len(message) > MAX_LOG_STRING_LEN:
                message = message[:MAX_LOG_STRING_LEN]
            invalid_chars = {
                '\f', '\r', '\b', '\t', '\v', '\n',
                '\u000A', '\u000D', '\u000C', '\u000B',
                '\u0008', '\u007F', '\u0009'
            }
            for char in invalid_chars:
                message = message.replace(char, "")
            message = re.sub(R"[ ]+", " ", message) 
        else:
            message = f'log is None!'
        return message


def create_directory_with_permissions(real_log_path, permission) -> bool:
    if not real_log_path.endswith('/'):
        real_log_path += '/'

    path_parts = real_log_path.split('/')
    current_path = '/'
    for part in path_parts:
        if not part:
            continue
        current_path = os.path.join(current_path, part)
        if os.path.exists(current_path):
            continue
        try:
            os.makedirs(current_path, mode=permission, exist_ok=True)
        except Exception:
            logging.warning("Failed to create log directory.")
            return False
    return True


def init_logger():
    global logger
    log_level = str_to_loglevel(ENV.component_log_level)
    logger.setLevel(log_level)
    if ENV.disable_log:
        logger.disabled=True
        return logger
    
    if ENV.component_log_verbose in POSITIVE_BOOLEAN:
        formatter = LoggerFormatter(
            '%(asctime)s [%(process)d] [%(thread)d] [MindIE-SD] [%(levelname)s] %(filename)s:%(lineno)d: %(message)s'
        )
    else:
        formatter = LoggerFormatter(
            '%(asctime)s [%(levelname)s]: %(message)s'
        )

    if ENV.component_log_stdout in POSITIVE_BOOLEAN:
        print_handler = logging.StreamHandler(stream=sys.stdout)
        print_handler.setFormatter(formatter)
        print_handler.setLevel(log_level)
        logger.addHandler(print_handler)
    
    if ENV.component_log_to_file in POSITIVE_BOOLEAN:
        # check and standarlize the path
        log_base_path = ENV.mindie_log_path
        log_base_path = os.path.expanduser(log_base_path) # expand '~'
        # relative path
        if not log_base_path.startswith("/"):
            log_base_path = os.path.join(MINDIE_DEFAULTS_LOG_PATH, log_base_path)
            log_base_path = os.path.expanduser(log_base_path) # expand '~'
        
        real_log_path = ""
        if check_path(log_base_path):
            real_log_path = os.path.realpath(log_base_path)
        need_add_handler = False
        if real_log_path:
            debug_log_path = os.path.join(real_log_path, "debug")
            need_add_handler = create_directory_with_permissions(debug_log_path, PATH_OWNER_SHIP)
            if need_add_handler:
                file_handler = MindIELogFileHandler(debug_log_path,
                                                    max_file_num=ENV.rotate_max_file_num,
                                                    max_file_size=ENV.rotate_max_file_size*MB,
                                                    rotate_cycle_num=ENV.rotate_cycle_num,
                                                    rotate_cycle=ENV.rotate_cycle)
                file_handler.setFormatter(formatter)
                file_handler.setLevel(log_level)
                logger.addHandler(file_handler)
        else:
            logging.warning("The log file path is invalid or does not exist. The log cannot be saved!")
    logger.propagate = False


logger = logging.getLogger('mindie-sd')
init_logger()