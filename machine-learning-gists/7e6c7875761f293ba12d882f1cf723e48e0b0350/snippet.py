#!/usr/bin/env python
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @Author: Binyuan Hui
# @Lab of Machine Learning and Data Mining, TianJin University
# @Email: huybery@gmail.com
# @Date: 2018-10-26 15:32:34
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import print_function
from __future__ import absolute_import

import argparse
import os

import subprocess


class NvidiaSuper:
    NVIDIA_COMMAND = 'nvidia-smi'

    def __init__(self):
        self.source = None
        self.gpu_process = []

        self._get_source()
        self._get_process_pool()

    def _get_source(self):
        try:
            res = subprocess.Popen(
                self.NVIDIA_COMMAND,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
                # oncomment below if you use py2
                # encoding='utf-8'  
                )
            self.source = res.stdout.readlines()
        except:
            raise EnvironmentError('No GPU driver.')

    def _get_process_line(self):
        for idx, line in enumerate(self.source):
            if 'Processes' in line:
                return idx

    def _get_process_pool(self):
        idx_line = self._get_process_line() + 3
        for line in self.source[idx_line:]:
            if line.startswith('+-'):
                break
            if 'No running processes found' in line:
                return []
            info_lst = line.strip().split()
            idx_gpu = info_lst[1]
            pid = info_lst[2]
            s = self.ps_info(pid)
            s.append('\n')
            info = []
            info.append(idx_gpu)
            # user
            info.append(s[0])
            # pid
            info.append(s[1])
            # stat
            info.append(s[7])
            # start
            info.append(s[8])
            # time
            info.append(s[9])
            command = ' '.join(s[10:])
            info.append(command)
            self.gpu_process.append('\t'.join(info))
        return self.gpu_process

    @staticmethod
    def ps_info(pid):
        res = subprocess.Popen(
            'ps -u -p ' + pid,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            # oncomment below if you use py2            
            # encoding='utf-8'
            )
        return res.stdout.readlines()[1].split()

    def print_to(self):
        print(''.join(self.source))
        title = ['GPU', 'USER', 'PID', 'STAT', 'START', 'TIME', 'COMMAND']
        print('\t'.join(title))
        print(''.join(self.gpu_process))


if __name__ == '__main__':
    mnitor = NvidiaSuper()
    mnitor.print_to()