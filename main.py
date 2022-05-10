# You should not modify this part.
import pandas as pd
import numpy as np
from keras.models import load_model


def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()

def next_day(today):
    days_each_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    ymd = today.split(" ")
    y, m, d = ymd[0].split("-")
    d = int(d) + 1
    if (days_each_month[int(m) - 1] < d):
        d = 1
        m = str(int(m) + 1)
        if m == "13":
            m = "1"
            y = str(int(y) + 1)

    return (y + '-' + m + '-' + str(d))

def load(file_c, file_g):
    dfc = pd.read_csv(file_c)
    c = dfc["consumption"].values

    dfg = pd.read_csv(file_g)
    g = dfg["generation"].values

    date = next_day(dfc["time"].values[-1])
    return [c, g, date]

def normalize(x):
    max_ = np.max(x)
    min_ = np.min(x)
    x = (x - min_) / (max_ - min_)
    # y = (y - min_) / (max_ - min_)
    return [x, max_, min_]
    
def denormalize(x, max, min):
    return x * (max - min) + min

def consumption_pred(model, test_x):
    predict_y = []
    predict_y.append(model.predict(np.array([test_x])))
    return predict_y

def generation_pred(week_gen):
    hours_gen = np.zeros(24)
    for d in range(7):
        hours_gen += week_gen[d * 24: (d + 1) * 24]
    peak = hours_gen[12] / 7
    hours_gen = hours_gen / np.max(hours_gen) * peak
    return hours_gen    

def trade_pred(consume, generate, date):
    trade_decision = []
    for i in range(24):
        time_str = date + " " + str(i) + ":00:00"
        if generate[i] < 0.2:
            trade_decision.append([time_str, "buy", 2.5256, consume[i]])
        elif generate[i] - consume[i] > 0.2:
            trade_decision.append([time_str, "sell", 1.2628, generate[i] - consume[i]])
    return trade_decision

def output(path, data):

    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return


if __name__ == "__main__":
    args = config()

    # data = [["2018-01-01 00:00:00", "buy", 2.5, 3],
    #         ["2018-01-01 01:00:00", "sell", 3, 5]]
    c, g ,date = load(args.consumption, args.generation)
    test_c, max, min = normalize(c)

    model = load_model("./lstm.hdf5")
    pred_c = consumption_pred(model, test_c)
    pred_c = denormalize(pred_c[0][0], max, min)
    pred_g = generation_pred(g)
    data = trade_pred(pred_c, pred_g, date)
    output(args.output, data)
