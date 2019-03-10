# Hsiaofei Tsien

import numpy as np
import pandas as pd
import scipy

def auto_bin(DF, X, Y, n=5, iv=True, detail=False,q=20):
    """
    自动最优分箱函数，基于卡方检验的分箱

    参数：
    DF: DataFrame 数据框
    X: 需要分箱的列名
    Y: 分箱数据对应的标签 Y 列名
    n: 保留分箱个数
    iv: 是否输出执行过程中的 IV 值
    detail: 是否输出合并的细节信息
    q: 初始分箱的个数

    区间为前开后闭 (]

    返回值：

    """


    # DF = model_data
    # X = "age"
    # Y = "SeriousDlqin2yrs"

    DF = DF[[X,Y]].copy()

    # 按照等频对需要分箱的列进行分箱
    DF["qcut"],bins = pd.qcut(DF[X], retbins=True, q=q, duplicates="drop")
    # 统计每个分段 0，1的数量
    coount_y0 = DF.loc[DF[Y]==0].groupby(by="qcut")[Y].count()
    coount_y1 = DF.loc[DF[Y]==1].groupby(by="qcut")[Y].count()
    # num_bins值分别为每个区间的上界，下界，0的频次，1的频次
    num_bins = [*zip(bins,bins[1:],coount_y0,coount_y1)]

    # 定义计算 woe 的函数
    def get_woe(num_bins):
        # 通过 num_bins 数据计算 woe
        columns = ["min","max","count_0","count_1"]
        df = pd.DataFrame(num_bins,columns=columns)

        df["total"] = df.count_0 + df.count_1
        df["percentage"] = df.total / df.total.sum()
        df["bad_rate"] = df.count_1 / df.total
        df["woe"] = np.log((df.count_0/df.count_0.sum()) /
                           (df.count_1/df.count_1.sum()))
        return df

    # 创建计算 woe 值函数
    def get_iv(bins_df):
        rate = ((bins_df.count_0/bins_df.count_0.sum()) -
                (bins_df.count_1/bins_df.count_1.sum()))
        woe = np.sum(rate * bins_df.woe)
        return woe


    # 确保每个分组的数据都包含有 0 和 1
    for i in range(20): # 初始分组不会超过20
        # 如果是第一个组没有 0 或 1，向后合并
        if 0 in num_bins[0][2:]:
            num_bins[0:2] = [(
                num_bins[0][0],
                num_bins[1][1],
                num_bins[0][2]+num_bins[1][2],
                num_bins[0][3]+num_bins[1][3])]
            continue

        # 其他组出现没有 0 或 1，向前合并
        for i in range(len(num_bins)):
            if 0 in num_bins[i][2:]:
                num_bins[i-1:i+1] = [(
                    num_bins[i-1][0],
                    num_bins[i][1],
                    num_bins[i-1][2]+num_bins[i][2],
                    num_bins[i-1][3]+num_bins[i][3])]
                break
        # 循环结束都没有出现则提前结束外圈循环
        else:
            break

    # 重复执行循环至分箱保留 n 组：
    while len(num_bins) > n:
        # 获取 num_bins 两两之间的卡方检验的置信度（或卡方值）
        pvs = []
        for i in range(len(num_bins)-1):
            x1 = num_bins[i][2:]
            x2 = num_bins[i+1][2:]
            # 0 返回 chi2 值，1 返回 p 值。
            pv = scipy.stats.chi2_contingency([x1,x2])[1]
            # chi2 = scipy.stats.chi2_contingency([x1,x2])[0]
            pvs.append(pv)

        # 通过 p 值进行处理。合并 p 值最大的两组
        i = pvs.index(max(pvs))
        num_bins[i:i+2] = [(
            num_bins[i][0],
            num_bins[i+1][1],
            num_bins[i][2]+num_bins[i+1][2],
            num_bins[i][3]+num_bins[i+1][3])]

        # 打印合并后的分箱信息
        bins_df = get_woe(num_bins)
        if iv:
            print(f"{X} 分{len(num_bins):2}组 IV 值: ",get_iv(bins_df))
        if detail:
            print(bins_df)
    # print("\n".join(map(lambda x:f"{x:.16f}",pvs)))
    # 返回分组后的信息
    return get_woe(num_bins) #, get_iv(bins_df)