import os

import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import plot

from pyecharts.charts import Kline, Line, Bar, Grid
from pyecharts import options as opts
from pyecharts.globals import ThemeType


def draw_equity_curve_mat_new(df, trade, title, path='./pic_new.html', show=False):
    """
    绘制附带K线的资金曲线和成交量
    :param df: 包含资金曲线列和成交量的df
    :param trade: 每笔交易
    :param title: 表名
    :param path: 保存路径
    :param show: 是否展示图片
    :return:
    """
    g = trade.copy()
    mark_point_list = []
    for i in g.index:
        buy_time = df[df['candle_begin_time'] == i].index[0]
        sell_time = df[df['candle_begin_time'] == g.loc[i, 'end_bar']].index[0]
        # 标记买卖点
        y_buy = df.loc[buy_time, 'high']
        y_sell = df.loc[sell_time, 'low']

        # 开仓点
        mark_point_list.append({
            'coord': [df['candle_begin_time'][buy_time], y_buy],
            'value': '开空' if g.loc[i, 'signal'] == -1 else '开多',
            'itemStyle': {'color': 'blue' if g.loc[i, 'signal'] == -1 else 'red'}
        })

        # 平仓点
        mark_point_list.append({
            'coord': [df['candle_begin_time'][sell_time], y_sell],
            'value': '平仓',
            'itemStyle': {'color': 'lightblue'}
        })

    # 创建K线图
    kline = Kline(init_opts=opts.InitOpts(theme=ThemeType.DARK, bg_color='transparent'))
    kline.add_xaxis(df['candle_begin_time'].tolist())
    kline.add_yaxis(
        "K线",
        df[['open', 'close', 'low', 'high']].values.tolist(),
        itemstyle_opts=opts.ItemStyleOpts(color="#ec0000", color0="#00da3c"),
        markpoint_opts=opts.MarkPointOpts(data=mark_point_list)
    )

    kline.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="category",
            is_scale=True,
            grid_index=0,
            splitline_opts=opts.SplitLineOpts(is_show=False),
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            is_scale=True,
            grid_index=0,
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
    )

    # 创建成交量柱状图
    volume = Bar()
    volume.add_xaxis(df['candle_begin_time'].tolist())
    volume.add_yaxis("成交量", df['quote_volume'].tolist(), label_opts=opts.LabelOpts(is_show=False),
                     itemstyle_opts=opts.ItemStyleOpts(color="lightblue"))

    volume.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="category",
            is_scale=True,
            grid_index=1,
            splitline_opts=opts.SplitLineOpts(is_show=False),
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            is_scale=True,
            grid_index=1,
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        title_opts=opts.TitleOpts(is_show=False),  # 隐藏标题
        legend_opts=opts.LegendOpts(pos_right="20%", pos_top="0%")  # 设置图例在右侧
    )
    # 创建资金曲线图
    line = Line()
    line.add_xaxis(df['candle_begin_time'].tolist())
    line.add_yaxis("资金曲线", df['equity_curve'].tolist(), linestyle_opts=opts.LineStyleOpts(color="red"),
                   label_opts=opts.LabelOpts(is_show=False))

    line.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="category",
            is_scale=True,
            grid_index=2,
            splitline_opts=opts.SplitLineOpts(is_show=False),
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            is_scale=True,
            grid_index=2,
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        title_opts=opts.TitleOpts(is_show=False),  # 隐藏标题
        legend_opts=opts.LegendOpts(pos_right="10%", pos_top="0%")  # 设置图例在右侧
    )

    # 创建布林轨道线
    line_bulin = Line()
    line_bulin.add_xaxis(df['candle_begin_time'].tolist())
    line_bulin.add_yaxis("上轨", df['upper'].tolist(), linestyle_opts=opts.LineStyleOpts(type_="dashed", color="blue"),
                         label_opts=opts.LabelOpts(is_show=False))  # 隐藏上轨线的数值
    line_bulin.add_yaxis("中轨", df['median'].tolist(),
                         linestyle_opts=opts.LineStyleOpts(type_="dashed", color="black"),
                         label_opts=opts.LabelOpts(is_show=False))  # 隐藏中轨线的数值
    line_bulin.add_yaxis("下轨", df['lower'].tolist(), linestyle_opts=opts.LineStyleOpts(type_="dashed", color="blue"),
                         label_opts=opts.LabelOpts(is_show=False))  # 隐藏下轨线的数值

    line_bulin.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="category",
            is_scale=True,
            grid_index=2,
            splitline_opts=opts.SplitLineOpts(is_show=False),
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            is_scale=True,
            grid_index=2,
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
    )

    # 共享的滑动条DataZoom
    datazoom_opts = opts.DataZoomOpts(
        is_show=True,
        type_="inside",
        xaxis_index=[0, 1, 2],  # 绑定三个图的x轴
        range_start=0,  # 显示数据的起始百分比
        range_end=100,  # 显示数据的结束百分比
    )

    overlap = kline.overlap(line_bulin)

    overlap.set_global_opts(datazoom_opts=[datazoom_opts])
    volume.set_global_opts(datazoom_opts=[datazoom_opts])
    line.set_global_opts(datazoom_opts=[datazoom_opts])

    # 使用Grid布局，将三个图表垂直排列
    grid = Grid(init_opts=opts.InitOpts(width="1400px", height="700px"))  # 增加高度以适应三个图表
    grid.add(overlap, grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", height="45%"))
    grid.add(volume, grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", pos_top="59%", height="12%"))
    grid.add(line, grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", pos_top="75%", height="15%"))

    grid.render(path)
    if show:
        grid.render_notebook()


def draw_chart_mat(df, draw_chart_list, pic_size=[9, 9], dpi=72, font_size=20, noise_pct=0.05):
    """
    绘制分布图
    :param df:  包含绘制指定分布数据的df
    :param draw_chart_list: 指定绘制的列
    :param pic_size:    指定画布大小
    :param dpi: 指定画布的dpi
    :param font_size:   指定字体大小
    :param noise_pct:   指定去除的异常值
    :return:
    """
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(num=1, figsize=(pic_size[0], pic_size[1]), dpi=dpi)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    row = len(draw_chart_list)
    count = 0
    for data in draw_chart_list:
        temp = df.copy()
        temp['Rank'] = temp[data].rank(pct=True)
        temp = temp[temp['Rank'] < (1 - noise_pct)]
        temp = temp[temp['Rank'] > noise_pct]
        # group = temp.groupby(data)
        # plt.hist(group.groups.keys(), 20)

        ax = plt.subplot2grid((row, 1), (count, 0))
        ax.hist(temp[data], 70)
        ax.set_xlabel(data)
        ax.set_ylabel('数量')
        count += 1

    plt.show()