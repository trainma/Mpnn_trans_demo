from geopy.distance import geodesic
import pandas as pd
import json
stations = {
    'Dongsi': [116.4246, 39.93041,0],
    'Nongzhanguan': [116.4647, 39.94146,1],
    'Tiantan': [116.4067, 39.88241,2],
    'Wanliu': [116.2982, 39.95975,3],
    'Changping': [116.2202, 40.22293,4],
    'Dingling': [116.2234, 40.29596,5],
    'Guanyuan': [116.3623, 39.93269,6],
    'Huairou': [116.6338, 40.31929,7],
    'Wanshouxigong': [116.3677, 39.88022,8],
    'Aotizhongxing': [116.3995, 39.98365,9],
    'Gucheng': [116.1933, 39.91163,10],
    'Shunyi': [116.6636, 40.13613,11]

}

print(geodesic((30.28708, 120.12802999999997), (28.7427, 115.86572000000001)).m)  # 计算两个坐标直线距离
print(geodesic((30.28708, 120.12802999999997), (28.7427, 115.86572000000001)).km)  # 计算两个坐标直线距离

if __name__ == '__main__':

    # for station in stations:
    #     for Next_station in stations:
    #         if station == Next_station:
    #             continue
    #         else:
    #             dist = geodesic((stations[station][1], stations[station][0]),
    #                             (stations[Next_station][1], stations[Next_station][0])).km
    #             print("{},{},{}".format(station, Next_station, dist))
    #             with open('dist.txt', 'a') as fp:
    #                 fp.writelines("{},{},{}\n".format(station, Next_station, dist))
    #df=pd.read_csv('dist.txt',sep=',')
    #df.to_csv('dist.csv')
    with open('Airquality/station.json', 'a') as fp:
        json.dump(stations,fp)