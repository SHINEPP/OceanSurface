# -*- coding: utf-8 -*-

import math
import numpy as np
from scipy.fftpack import fft


class OceanSurface:
    def __init__(self):
        self.__g = 9.8
        self.__l = 0.1
        self.__A = 3.0e-7

        self.__lambda = 1.0

        self.__V = 30  # 风速
        self.__omega_hat = [1, 1]  # 风向

        # Mesh resolution
        self.__N = 64
        self.__M = 64
        self.__L_x = 1000.0
        self.__L_z = 1000.0

        np.random.seed(0)

        self.__kNum = self.__N * self.__M

        self.height_field = []
        self.__value_h_twiddle_0 = []
        self.__value_h_twiddle_0_conj = []
        self.__value_h_twiddle = []
        for i in range(0, self.__kNum):
            self.height_field.append([0.0, 0.0, 0.0])
            self.__value_h_twiddle_0.append(complex(0, 0))
            self.__value_h_twiddle_0_conj.append(complex(0, 0))
            self.__value_h_twiddle.append(complex(0, 0))

        for n in range(0, self.__N):
            for m in range(0, self.__M):
                index = m * self.__N + n
                k = self.__k_vec(n, m)
                self.__value_h_twiddle_0[index] = self.__func_h_twiddle_0(k)
                self.__value_h_twiddle_0_conj[index] = np.conj(self.__func_h_twiddle_0(k))

    def get_size(self):
        return self.__M, self.__N

    def build_field(self, time):
        slope_x_term = []
        slope_z_term = []
        d_x_term = []
        d_z_term = []

        for n in range(0, self.__N):
            for m in range(0, self.__M):
                index = m * self.__N + n

                self.__value_h_twiddle[index] = self.__func_h_twiddle(n, m, time)

                kVec = self.__k_vec(n, m)
                kLength = math.sqrt(kVec[0] * kVec[0] + kVec[1] * kVec[1])
                kVecNormalized = kVec
                if kLength != 0:
                    kVecNormalized = [kVec[0] / kLength, kVec[1] / kLength]

                slope_x_term.append(complex(0.0, kVec[0]) * self.__value_h_twiddle[index])
                slope_z_term.append(complex(0.0, kVec[1]) * self.__value_h_twiddle[index])
                d_x_term.append(complex(0.0, -kVecNormalized[0]) * self.__value_h_twiddle[index])
                d_z_term.append(complex(0.0, -kVecNormalized[1]) * self.__value_h_twiddle[index])

        out_height = fft(self.__value_h_twiddle)
        out_d_x = fft(d_x_term)
        out_d_z = fft(d_z_term)

        for n in range(0, self.__N):
            for m in range(0, self.__M):
                index = m * self.__N + n

                sign = 1
                if (m + n) % 2 != 0:
                    sign = -1

                self.height_field[index][0] = ((n - self.__N / 2) * self.__L_x / self.__N -
                                               sign * self.__lambda * out_d_x[index].imag)
                self.height_field[index][1] = sign * out_height[index].imag
                self.height_field[index][2] = ((m - self.__M / 2) * self.__L_z / self.__M -
                                               sign * self.__lambda * out_d_z[index].imag)

    def __func_omega(self, k):
        return math.sqrt(self.__g * k)

    def __func_P_h(self, vec_k):
        if vec_k[0] == 0.0 and vec_k[1] == 0.0:
            return 0.0

        L = self.__V * self.__V / self.__g
        k = math.sqrt(vec_k[0] * vec_k[0] + vec_k[1] * vec_k[1])
        k_hat = [vec_k[0] / k, vec_k[1] / k]

        dot_k_hat_omega_hat = k_hat[0] * self.__omega_hat[0] + k_hat[1] * self.__omega_hat[1]
        result = self.__A * math.exp(-1.0 / (k * L * k * L)) / math.pow(k, 4) * math.pow(dot_k_hat_omega_hat, 2)

        result *= math.exp(-k * k * self.__l * self.__l)
        return result

    def __func_h_twiddle_0(self, vec_k):
        xi = np.random.standard_normal(2)
        return math.sqrt(0.5) * complex(xi[0], xi[1]) * math.sqrt(self.__func_P_h(vec_k))

    def __func_h_twiddle(self, kn, km, t):
        index = km * self.__N + kn
        vec = self.__k_vec(kn, km)
        k = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
        term1 = self.__value_h_twiddle_0[index] * np.exp(complex(0.0, self.__func_omega(k) * t))
        term2 = self.__value_h_twiddle_0_conj[index] * np.exp(complex(0.0, -self.__func_omega(k) * t))
        return term1 + term2

    def __k_vec(self, n, m):
        k1 = 2 * math.pi * (n - self.__N / 2.0) / self.__L_x
        k2 = 2 * math.pi * (m - self.__M / 2.0) / self.__L_z
        return [k1, k2]
