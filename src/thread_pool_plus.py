# Copyright (c) 2023 Yanhao Li
# yanhao.li@outlook.com

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


from typing import List
import sys
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime


class ThreadPoolPlus:
    lock = Lock()

    def __init__(self, workers) -> None:
        self.tasks_total = 0
        self.tasks_completed = 0
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.futures: List[Future] = []

        self.start_time = datetime.now()

    def progress_indicator(self, future):
        # obtain the lock
        with ThreadPoolPlus.lock:
            # update the counter
            self.tasks_completed += 1
            time_used = datetime.now() - self.start_time
            time_remain = (
                time_used
                / self.tasks_completed
                * (self.tasks_total - self.tasks_completed)
            )

            ThreadPoolPlus._progress_bar(
                self.tasks_completed, self.tasks_total, time_used, time_remain
            )

    def submit(self, task, *args):
        self.tasks_total += 1
        future:Future = self.pool.submit(task, *args)
        future.add_done_callback(self.progress_indicator)
        self.futures.append(future)

    def empty(self):
        return len(self.futures) == 0

    def pop_result(self):
        return self.futures.pop(0).result()

    def join(self):
        for future in self.futures:
            future.result()

    def stop(self):
        for future in self.futures:
            future.cancel()

    @staticmethod
    def _progress_bar(count_value: int, total: int, time_pass, time_remain):
        bar_length = 50
        filled_up_Length = int(round(bar_length * count_value / float(total)))
        percentage = int(100.0 * count_value / float(total))
        time_pass = ThreadPoolPlus.timedelta2string(time_pass)
        time_remain = ThreadPoolPlus.timedelta2string(time_remain)
        status = "%s/%s" % (count_value, total) + " [%s<%s]" % (time_pass, time_remain)
        bar = "#" * filled_up_Length + " " * (bar_length - filled_up_Length)
        if count_value < total:
            end = "\r"
        else:
            end = "\n"
        out_content = "%s%% |%s| %s %s" % (percentage, bar, status, end)
        sys.stdout.write(out_content)
        sys.stdout.flush()

    @staticmethod
    def timedelta2string(tdelta) -> str:
        hours, rem = divmod(tdelta.seconds, 3600)
        minutes, rem = divmod(rem, 60)
        seconds = rem
        return "%d:%02d:%02d" % (hours, minutes, seconds)
