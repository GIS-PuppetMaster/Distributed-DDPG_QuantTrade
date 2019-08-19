from datetime import *
import glo
from jqdatasdk import *
import numpy as np
import pandas as pd
import random


class StockDate:
    def __init__(self, stock_code):
        # np[str]
        self.stock_code = stock_code
        self.date_list = np.array(
            pd.read_csv('Data/' + self.stock_code.replace(".", "_") + ".csv")['Unnamed: 0'])
        # datetime
        self.date = datetime.strptime(self.date_list[0], "%Y-%m-%d %H:%M:%S")
        self.index = 0

    def set_date(self, date=None):
        # 输入datetime
        if date is None:
            # 随机初始化日期
            self.date = datetime.strptime(
                self.date_list[random.randint(glo.day * 300 + 1, len(self.date_list) - 365 * 300 - 1)],
                "%Y-%m-%d %H:%M:%S")
        else:
            self.date = date
        self.index = self.find_index(0, self.date_list.size - 1)

    def set_date_with_index(self, date, index):
        self.date = date
        self.index = index

    def next_date(self):
        frequency = int(glo.frequency[:-1])
        self.date = datetime.strptime(self.date_list[self.index + frequency], "%Y-%m-%d %H:%M:%S")
        self.index += frequency
        return self.date

    def last_date(self):
        frequency = int(glo.frequency[:-1])
        self.date = datetime.strptime(self.date_list[self.index - frequency], "%Y-%m-%d %H:%M:%S")
        self.index -= frequency
        return self.date

    def next_day(self):
        day = self.date.day
        # 如果日期没有变更则next_date
        while self.date.day == day:
            self.next_date()
        return self.date

    def last_day(self):
        # 将时间调至前一天同一时间
        self.date = self.date - timedelta(days=1)
        i = self.find_index(0, self.date_list.size - 1)
        while i is None:
            self.date = self.date - timedelta(days=1)
            i = self.find_index(0, self.date_list.size - 1)
        self.index = i
        return self.date

    def get_date(self):
        return self.date

    def find_index(self, start, end):
        """
        二分查找当前日期对应的索引
        :param start: 查找起始索引
        :param end: 查找终止索引
        :return: 如果找到则返回索引，否则返回None
        """
        if start <= end:
            mid = int((start + end) / 2)
            if datetime.strptime(self.date_list[mid], "%Y-%m-%d %H:%M:%S").__gt__(self.date):
                return self.find_index(start, mid - 1)
            elif datetime.strptime(self.date_list[mid], "%Y-%m-%d %H:%M:%S").__lt__(self.date):
                return self.find_index(mid + 1, end)
            return mid
        return None

    def get_index(self):
        return self.index
