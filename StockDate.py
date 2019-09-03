from datetime import *
import glo
from jqdatasdk import *
import random


class StockDate:
    def __init__(self, stock_code):
        # np[str]
        self.stock_code = stock_code
        self.date_list = glo.date[stock_code]
        # datetime
        self.date = datetime.strptime(self.date_list[0], "%Y-%m-%d %H:%M:%S")
        self.index = 0
        self.dict = glo.dict[stock_code]

    def set_date(self, date=None):
        # 输入datetime
        if date is None:
            # 随机初始化日期
            # 前一段时间为保留日期不可设置
            self.date = datetime.strptime(
                self.date_list[random.randint(2 * glo.day * 300 + 1, len(self.date_list) - 1)],
                "%Y-%m-%d %H:%M:%S")
        else:
            self.date = date
        # self.index = self.find_index(0, self.date_list.size - 1)
        self.index = self.find_index()

    def set_date_with_index(self, date, index):
        self.date = date
        self.index = index

    def next_date(self):
        frequency = int(glo.frequency[:-1])
        next_index = self.index + frequency
        self.date = datetime.strptime(self.date_list[next_index], "%Y-%m-%d %H:%M:%S")
        self.index = next_index
        # 如果下一步溢出则返回None
        if self.index + frequency >= len(self.date_list) or self.index == len(self.date_list) - 1:
            return self.date, True
        return self.date, False

    def last_date(self):
        frequency = int(glo.frequency[:-1])
        self.date = datetime.strptime(self.date_list[self.index - frequency], "%Y-%m-%d %H:%M:%S")
        self.index -= frequency
        return self.date

    def next_day(self):
        day = self.date.day
        # 如果日期没有变更则next_date
        over_flow = False
        while self.date.day == day and not over_flow:
            d, over_flow = self.next_date()
        return self.date, over_flow

    def last_day(self):
        # 将时间调至前一天同一时间
        self.date = self.date - timedelta(days=1)
        i = self.find_index()
        while i is None:
            self.date = self.date - timedelta(days=1)
            i = self.find_index()
        self.index = i
        return self.date

    def get_date(self):
        return self.date

    def find_index(self):
        if str(self.date) in self.dict.keys():
            return self.dict[str(self.date)]
        else:
            return None

    def get_index(self):
        return self.index
