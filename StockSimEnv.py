from sklearn.preprocessing import *
from StockDate import *
import numpy as np


# auth("13074581737", "trustno1")


class Env:
    def __init__(self, stock_code=None, start_date=None, ori_money=pow(10, 6), quant=None):
        """
        初始化环境
        :return: None
        """
        # 未指定股票代码则随机选择股票代码
        if stock_code is None:
            self.stock_code = glo.random_stock_code()
        else:
            self.stock_code = stock_code
        # 默认随机初始化时间
        self.gdate = StockDate(self.stock_code)
        self.gdate.set_date(start_date)
        self.temp_date = StockDate(self.stock_code)
        self.temp_date.date = self.gdate.date
        self.temp_date.index = self.gdate.index
        self.ori_money = ori_money
        self.ori_value = 0
        self.money = self.ori_money
        self.stock_value = []
        self.data = glo.data[self.stock_code]
        self.price = self.get_stock_price(self.gdate.get_date())
        self.init_with_ori_stock_value(self.price, quant)
        self.time_list = []
        self.profit_list = []
        self.reference_list = []
        self.price_list = []
        self.start_date = self.gdate.get_date()

    def reset(self, start_date=None, ori_money=pow(10, 6), quant=None):
        self.gdate = StockDate(self.stock_code)
        self.gdate.set_date(start_date)
        self.temp_date = StockDate(self.stock_code)
        self.temp_date.date = self.gdate.date
        self.temp_date.index = self.gdate.index
        self.ori_money = ori_money
        self.ori_value = 0
        self.money = self.ori_money
        self.stock_value = []
        self.price = self.get_stock_price(self.gdate.get_date())
        self.init_with_ori_stock_value(self.price, quant)
        self.time_list = []
        self.profit_list = []
        self.reference_list = []
        self.price_list = []
        self.start_date = self.gdate.get_date()
        print("重置环境完成")

    def init_with_ori_stock_value(self, price, quant=None):
        temp_quant = quant
        if temp_quant is None:
            # 初始化时默认持股
            amount = int(self.money / (100 * price))
            # 实际买多少手
            temp_quant = int(1 * amount)
        self.stock_value = [[price, temp_quant]]
        self.ori_value = price * temp_quant * 100

    def get_stock_total_value(self, price=-1):
        """
           计算股票总价值
           price默认值为self.price
           :return: 股票总价值
        """
        if price == -1:
            price = self.price
        return price * 100 * self.get_stock_amount()

    def get_stock_amount(self):
        """返回当前持有的股票总数目"""
        amount = 0
        for l in self.stock_value:
            amount += l[1]
        return amount

    def get_state(self, date=None):
        """
        获取指定时间的状态
        :param date: 指定的时间，默认为当前时间
        :return: 指定时间的stock和agent状态，已经做完必要的数据处理，可直接输入到神经网使用
        """
        """
        stock_state = np.array(
            get_price(glo.stock_code, count=glo.count, frequency=glo.frequency,
                      end_date=date.strftime("%Y-%m-%d %H:%M:%S"),
                      skip_paused=True))[..., [0, 1, 2, 3]].transpose().tolist()
        """
        if date is None:
            self.temp_date.set_date(self.gdate.get_date())
        else:
            self.temp_date.set_date(date)
        stock_state = []
        # 获取前5天的数据
        for i in range(0, glo.day):
            day_state = []
            # 获取前count分钟的数据
            for j in range(0, glo.count):
                line = np.array(self.data.loc[str(self.temp_date.get_date())])[[0, 1, 2, 3, 4, 5]].transpose().tolist()
                self.temp_date.last_date()
                day_state.append(line)
            # 归一化
            day_state = MinMaxScaler().fit_transform(scale(day_state, axis=0))
            stock_state.append(day_state)
            # 前一天
            self.temp_date.last_day()
        # 生成agent状态
        agent_state = [self.money] + [self.get_stock_total_value(self.price)] + [
            self.get_stock_amount()]
        agent_state = np.array(agent_state).reshape(-1, 1).tolist()
        # 归一化
        agent_state = MinMaxScaler().fit_transform(scale(agent_state, axis=0))
        # 整理维度
        stock_state = np.array(stock_state).transpose((1, 0, 2)).reshape(1, glo.count, glo.day,
                                                                         glo.stock_state_size).tolist()
        # stock_state = np.array(stock_state).reshape(1, glo.day, glo.stock_state_size, glo.count).tolist()
        agent_state = np.array(agent_state).reshape(1, glo.agent_state_size).tolist()
        return stock_state, agent_state

    def get_stock_price(self, date=None):
        """
        获取指定时间的股价
        :param date: 指定时间，默认设置当前时间
        :return:
        """
        """
        return np.array(get_price(glo.stock_code, count=1, frequency=glo.frequency,
                                  end_date=date.strftime("%Y-%m-%d %H:%M:%S"),
                                  skip_paused=True)).tolist()[0][1]
        """
        return self.get_single_stock_state(date)[1]

    def get_single_stock_state(self, date=None):
        if date is None:
            self.temp_date.set_date(self.gdate.get_date())
        else:
            self.temp_date.set_date(date)
        return np.array(self.data.loc[str(self.temp_date.get_date())]).tolist()

    def trade(self, action):
        """
        交易函数
        :param action:动作
        :return: 下一状态的stock_state,agent_state,reward,本次交易量
        """
        """action:买入股票,stock股票代码,quant卖出手数"""
        self.temp_date.set_date(self.gdate.get_date())
        quant = 0
        flag = False
        action_0 = action[0]
        self.price = self.get_stock_price()
        # 交易开关激活时，计算quant，否则quant=0
        if action[1] > 0:
            # 买入
            if action_0 > 0:
                # 按钱数百分比买入
                # 当前的钱购买多少手
                amount = int(self.money / (100 * self.price))
                # 实际买多少手
                quant = int(action_0 * amount)
                if quant == 0:
                    print("钱数：" + str(self.money) + "不足以购买一手股票")
                    flag = True
            # 卖出
            elif action_0 < 0:
                # 当前手中有多少手
                amount = self.get_stock_amount()
                if amount == 0:
                    flag = True
                # 实际卖出多少手
                quant = int(action_0 * amount)
                if quant == 0 and action_0 != 0:
                    flag = True
        # 获取当前价格
        price = self.price
        # 钱数-=每股价格*100*交易手数
        self.money = self.money - price * 100 * quant - abs(price * 100 * quant * 1.25 / 1000)
        if quant != 0:
            # 如果实际交易了则记录
            # [股价,手数]
            self.stock_value.append([price, quant])
        """
        print("日期:" + str(self.gdate.get_date()))
        print("action:" + str(action))
        print("实际交易量：" + str(quant))
        print("实际交易金额（收入为正卖出为负）：" + str(-price * 100 * quant))
        """
        # 更新环境时间片段
        # 如果没交易则下一时刻
        if quant == 0:
            next_date, over_flow = self.gdate.next_date()
        # 如果交易了则跳到下一天
        else:
            # 记录交易日期
            self.time_list.append(self.gdate.get_date())
            # 记录策略利率和基准利率
            self.profit_list.append(
                (self.get_stock_total_value(self.price) + self.money - self.ori_money - self.ori_value) / (
                        self.ori_value + self.ori_money))
            self.reference_list.append(
                ((self.price * 100 * self.stock_value[0][1] - self.ori_value) / (self.ori_value + self.ori_money)))
            # 记录股价
            self.price_list.append(self.price)
            next_date, over_flow = self.gdate.next_day()
        # 计算reward
        if flag:
            # 惩罚
            reward = -abs(action_0)
        else:
            reward = (self.money + self.get_stock_total_value(
                self.get_stock_price(next_date)) - self.ori_money - self.ori_value) / (self.ori_money + self.ori_value)

        # 交易了返回下一天的状态，否则返回下一个frequency的状态
        state = self.get_state(date=next_date)
        if self.gdate.index == len(self.gdate.date_list) - 1 or over_flow or (next_date - self.start_date).days >= 365:
            reward=None
        return state[0], state[1], reward
