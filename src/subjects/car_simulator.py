# -*- coding: utf-8 -*-
# Copyright 2018 Event Vision Library.

import numpy as np
from scipy import interpolate

from subjects import SubjectBase

"""
External constants
"""

c_d = 0.4               # 空気抵抗係数
wind_velocity = np.zeros(3)     # 風の向きベクトル [m/s]
temparature = 25        # 気温 [℃]
air_pressure = 1.0      # 大気圧 [atm]
water_vapor_pressure = 0.0      # 水蒸気圧 [atm]
rho_air = 1.293 * air_pressure * (1 - 0.378 * water_vapor_pressure / air_pressure) / (1 + temparature / 273.15)  # 温度に従って空気の密度 [kg/m3]（通常1.293程度）
mu_r = 0.1              # 路面の転がり抵抗係数
vec_r = np.array([0.0, 0.0, 1.0])   # 路面の法線ベクトル
g = 9.8                 # 重力加速度 [m/s2]
mu_s = 0.7              # スキッド限界の定数

Weight = 1500.0        # 質量 [kg]
Wheelbase = 2.5        # ホイールベース [m]
Height_Center = 0.6    # 重心の高さ [m]
X = np.linspace(1000, 8000, 15)
Y = np.array([135., 150., 162, 180, 185,
              200, 196, 193, 190, 182,
              178, 174, 165, 158, 140])
Tolcfunc = interpolate.interp1d(X, Y)  # エンジンのトルク関数（テーブル） [N・m]


class Car(object):
    """
    For Simulating Car movements
    see http://dskjal.com/programming/car-physics-for-simulator.html
    """
    def __init__(self, position=np.zeros(3), velocity=np.zeros(3), dir_front=0.0):

    def __init__(self, mass=1.0):
        self.action_list = ['upward', 'downward', 'rotate', 'stop']
        super(SimpleWalker, self).__init__(mass)

        """ 車オブジェクトの定数 """
        self.l_r = np.float32(1.5)              # 重心から後輪までの長さ [m]
        self.height = np.float32(1.5)           # 高さ [m]
        self.tradwidth = np.float32(1.5)        # トレッド幅 [m]
        self.wheelradius = np.float32(0.4)      # タイヤの半径 [m]
        self.geerratiotable = np.array([3.353, 3.846, 2.385, 1.750, 1.333, 1.042, 0.815, 0.0])  # ギア比テーブル
        self.defratio = np.float32(4.4)         # デフ比
        self.maxrad = np.pi / np.float32(4.0)   # タイヤの最大舵角 [rad]
        self.skidlimit = Weight * mu_s     # タイヤのスキッド限界
        self.maxbrakepower = np.float32(5000)   # 最大制動力[N]

        """計算のスピードアップのために先に計算しておく定数"""
        self.A = self.tradwidth * self.height

        """ 車オブジェクトの状態変数 """
        self.pitch_rotate = np.float32(0.0)     # 前後方向の車体の傾き
        self.roll_rotate = np.float32(0.0)      # 左右方向の車体の傾き
        self.steeringangle = np.float32(0.0)
        self.position = position    # 位置ベクトル[m]
        self.velocity = velocity    # 速度ベクトル[m/s]
        self.squared_velocity = self.velocity ** 2
        self.dir_front = dir_front  # 現在の前輪の向き [度]
        self.rad_front = dir_front * np.pi / np.float32(180.0)      # 現在の前輪の向き [rad]
        self.rotational_speed = self.calcEngineRotation()   # エンジンの回転数 [rpm]

        # 1フレームの間にステアを非現実的な速度で操作できないよう、現在のハンドルの回転度合を保存
        self.rotational_now = np.zeros(1)                   # ハンドルの回転度合 [rad]

    def update(self, accerelate, brake, steeringangle, dt=0.01):
        """ ユーザーからの入力：アクセルの踏み込み具合、ブレーキの踏み込み具合、ステアリングアングル　"""
        # steeringangle = np.clip(steeringangle, self.rotational_now - 2.0, self.rotational_now + 2.0)[0]
        # self.rotational_now += steeringangle
        braking = accerelate - brake
        self.velocity_before = self.velocity.copy()
        self.steeringangle = steeringangle
        if self.rotational_speed > 7000:
            braking = np.clip(braking, -100, 0.0)
        elif self.rotational_speed < 50 and braking < 0:
            braking = np.clip(braking, 0.0, 100.0)
        if self.rotational_speed > 50:           # and braking <= 0):
            self.dir_front += steeringangle
            self.dir_front = self.dir_front % 360.0
            self.rad_front += steeringangle * np.pi / 180.0

        if self.rotational_speed < 50 and braking > 0:
            self.rotational_speed = 1000
        self.velocity = self.calcVelocity(braking, steeringangle, dt=dt) # steeringangle, dt=dt)
        self.squared_velocity = self.velocity ** 2
        self.position = self.position + dt * self.velocity
        self.rotational_speed = self.calcEngineRotation()

    def getgeerratio(self):
        if np.sum(self.squared_velocity) < 3.0:
            return self.geerratiotable[2]
        else:
            return self.geerratiotable[3]

    # トラクションを計算
    def calcTraction(self, brake):
        u1 = np.array([np.cos(self.rad_front), np.sin(self.rad_front), 0.0])
        # u2 = self.velocity / np.linalg.norm(self.velocity)
        tolc = self.calcEngineTolc(self.rotational_speed * 1.02)
        # n = 0.9         # Transmission effectivity
        # r_t = 3.0          # 0.5       # throttle
        Ft = u1 * tolc * 3.0 * self.getgeerratio() * self.defratio * 0.9 / self.wheelradius
        return Ft

    # エンジンのトルクを計算
    def calcEngineTolc(self, rotational_speed):
        if rotational_speed <= 50:
            return 0.0
        elif 50 < rotational_speed < 300:
            return 60.0
        elif 300 < rotational_speed < 1000:
            return Tolcfunc(1000)
        elif rotational_speed > 8000:
            return Tolcfunc(8000)
        else:
            return Tolcfunc(rotational_speed)

    # 空気抵抗を計算
    def calcAirResistance(self):
        u = self.velocity / np.linalg.norm(self.velocity)
        if np.sum(self.squared_velocity) < 150.0:
            return np.zeros(3)
        else:
            Fair = - 1.0 * self.velocity * rho_air * c_d * self.A * np.sum((self.velocity - wind_velocity) ** 2) / 2.0
            return Fair

    # 転がり抵抗を計算
    def calcRotatingResistance(self, Ft, steeringangle, braking):
        u1 = -1.0 * np.array([np.cos(self.rad_front), np.sin(self.rad_front), 0.0])
        if np.all(Ft == 0.0):
            u2 = np.zeros(3)
        else:
            u2 = - 1.0 * Ft / np.linalg.norm(Ft)
        w_fo, w_fi, w_ro, w_ri = self.calcLoadofwheel(steeringangle, braking)
        Frr = u1 * mu_r * (w_fi + w_fo) + u2 * mu_r * (w_ri + w_ro)
        return Frr

    # タイヤにかかる荷重を計算
    def calcLoadofwheel(self, steeringangle, braking):
        R = self.calcRotationRadius(steeringangle)
        temp_f = (self.l_r - Height_Center * self.pitch_rotate) / Wheelbase - Height_Center * braking / g / Wheelbase / np.cos(self.pitch_rotate)
        temp_f2 = Height_Center * (self.roll_rotate - self.squared_velocity / g / R)
        w_fo = Weight * temp_f * (0.5 - temp_f2)
        w_fi = Weight * temp_f * (0.5 + temp_f2)
        w_ro = Weight * (1.0 - temp_f) * (0.5 - temp_f2)
        w_ri = Weight * (1.0 - temp_f) * (0.5 + temp_f2)
        return w_fo, w_fi, w_ro, w_ri

    # 旋回半径を計算
    def calcRotationRadius(self, steeringangle):
        steeringangle_rad = steeringangle * np.pi / 180.0
        return Wheelbase / np.sin(steeringangle_rad)

    # 遠心力を計算
    def calcCentrifugalForce(self, steeringangle):
        R = self.calcRotationRadius(steeringangle)
        Fc = Weight * self.squared_velocity / g / R
        return Fc

    # 制動力を計算
    def calcBrakeForce(self, brake, Ft):
        if np.all(Ft == 0.0):
            return np.zeros(3)
        else:
            return Ft / np.linalg.norm(Ft) * brake * self.maxbrakepower       # Fb

    # コーナリングパワーを計算
    def calcCorneringPower(self, steeringangle):
        steeringangle_rad = steeringangle * np.pi / 180.0
        direction_vec = np.array([np.cos(steeringangle_rad), np.sin(steeringangle_rad), 0.0])
        if np.linalg.norm(self.velocity_before) == 0.0 or steeringangle_rad == 0.0:
            return np.zeros(3)
        else:
            u_velo = self.velocity_before / np.linalg.norm(self.velocity_before)
            delta_deg = np.arccos(np.dot(direction_vec, u_velo)) * 180.0 / np.pi
            u = np.dot(u_velo[:2], np.array([[0.0, 1.0], [-1.0, 0.0]]))
            if delta_deg < 5:
                Fcn = u * 52 * delta_deg
            elif delta_deg < 10:
                Fcn = u * (12 * delta_deg + 200.0)
            else:
                Fcn = u * 330
            return np.append(Fcn * g, 0.0)

    def calcRad(self, F):
        return np.arctan(F[1] / F[0])

    # 車にかかる力を計算
    # def calcVarsbeforeForce(self, brake, steeringangle):
    #     self.v_u = self.velocity / np.linalg.norm(self.velocity)

    def calcForce(self, brake, steeringangle):
        # self.calcVarsbeforeForce(brake, steeringangle)
        Ft = self.calcTraction(brake)
        Fair = self.calcAirResistance()
        Frr = self.calcRotatingResistance(Ft, steeringangle, brake)
        Fc = self.calcCentrifugalForce(steeringangle)
        Fcn = self.calcCorneringPower(steeringangle)
        Fb = self.calcBrakeForce(brake, Ft)
        if np.sum(self.squared_velocity) < 100.0 and brake < 0:
            Fb = - 1.0 * Ft
        Fall = Ft + Fair + Frr + Fc + Fb + Fcn
        return Fall

    # 加速度を計算
    def calcAccerelation(self, brake, steeringangle):
        Fall = self.calcForce(brake, steeringangle)
        return Fall / Weight

    # 速度を計算
    def calcVelocity(self, brake, steeringangle, dt=0.01):
        if np.sum(self.squared_velocity) < 150.0 and brake < 0.1:
            return np.zeros(3)
        else:
            return self.velocity + dt * self.calcAccerelation(brake, steeringangle)

    # 位置を計算
    def calcPosition(self, brake, dt=0.01):
        return self.position + dt * self.calcVelocity(brake)

    # 速度からエンジンの回転数を再計算
    def calcEngineRotation(self):
        # slipratio = 1.02
        return 1.02 * np.linalg.norm(self.velocity) * 60.0 * self.getgeerratio() * self.defratio / (2 * np.pi * self.wheelradius)

    # 車体の傾きを計算          # 未実装
    # def calcPitchandRoll(self):
    #     return pitch_rotate, roll_rotate
