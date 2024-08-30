import os
import pandas as pd

os.chdir("/Users/utopia/Desktop/TIME")

# ===获取项目根目录
_ = os.path.abspath(os.path.dirname(__file__))  # 返回当文件路径
root_path = os.path.abspath(os.path.join(_, '..'))  # 返回根目录文件夹
print(_)
exit()

# ===通用配置
rule_types = ['8H']
symbol_list = ['XRP-USDT', 'TRX-USDT', "ETH-USDT"]  # 选择交易的币种列表
leverage_rate = 1  # 杠杆倍数
date_start = '2021-04-01'  # 回测开始时间
date_end = '2024-09-29'  # 回测结束时间
c_rate = 5 / 10000  # 手续费，commission fees，默认为万分之5。不同市场手续费的收取方法不同，对结果有影响。比如和股票就不一样。
slippage = 1 / 1000  # 滑点 ，可以用百分比，也可以用固定值。建议币圈用百分比，股票用固定值
min_margin_ratio = 1 / 100  # 最低保证金率，低于就会爆仓
drop_days = 10  # 币种刚刚上线10天内不交易
symbol_data_path = r'/Users/utopia/Downloads/coin-binance-swap-candle-csv-1h-2024-07-22'  # 永续合约一小时数据路径

# 最小下单量
min_amount_df = pd.read_csv(os.path.join(root_path, 'data/合约面值.csv'), encoding='gbk')
min_amount_dict = {}
for i in min_amount_df.index:
    min_amount_dict[min_amount_df.at[i, '合约']] = min_amount_df.at[i, '最小下单量']

# ===单策略回测配置
signal_name = 'signal_bollinger_squeeze_with_mfi_2'  # 策略名称
symbol = 'MATIC-USDT'  # 指定币种
para = [50, 2]  # 策略参数
proportion = 0.05  # 止盈止损比例


# ===遍历策略参数配置
strategies = ['signal_bollinger_squeeze_with_mfi', 'signal_bollinger_squeeze_with_mf_2']
