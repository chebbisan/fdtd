from math import floor
import numpy as np
import scipy.constants as constants

import tools

# Вариант 3

class GaussianModPlaneWave:
    ''' Класс с уравнением плоской волны для модулированного гауссова сигнала в дискретном виде
    d - определяет задержку сигнала.
    w - определяет ширину сигнала.
    Nl - количество ячеек на длину волны.
    Sc - число Куранта.
    eps - относительная диэлектрическая проницаемость среды, в которой расположен источник.
    mu - относительная магнитная проницаемость среды, в которой расположен источник.
    '''
    def __init__(self, d, w, Nl, Sc=1.0, eps=1.0, mu=1.0):
        self.d = d
        self.w = w
        self.Nl = Nl
        self.Sc = Sc
        self.eps = eps
        self.mu = mu

    def getE(self, m, q):
        '''
        Расчет поля E в дискретной точке пространства m
        в дискретный момент времени q
        '''
        return (np.sin(2 * np.pi / self.Nl * (q * self.Sc - m * np.sqrt(self.eps * self.mu))) *
                np.exp(-(((q - m * np.sqrt(self.eps * self.mu) / self.Sc) - self.d) / self.w) ** 2))

class Sampler:
    def __init__(self, discrete: float):
        self.discrete = discrete

    def sample(self, x: float) -> int:
        return floor(x / self.discrete + 0.5)


if __name__ == '__main__':
    # Init parameters

    # Волновое сопротивление свободного пространства
    W0 = 120.0 * np.pi
    # Число Куранта
    Sc = 1.0
    # Шаг сетки
    dx = 5e-3
    # Размер области моделирования в м
    maxSize_m = 1.5
    # Время расчета в секундах
    maxTime_s = 1e-8
    # частота в Гц
    freq_low = 3e9
    freq_high = 8e9

    frequency = (freq_low + freq_high) / 2
    # Положение источника в м
    sourcePos_m = maxSize_m / 3
    # Дискрет по времени
    dt = dx * Sc / constants.c
    # Инициализация семплеров
    sampler_x = Sampler(dx)
    sampler_t = Sampler(dt)
    # Время расчета в отсчетах
    maxTime = sampler_t.sample(maxTime_s)
    # Размер области моделирования в отсчетах
    maxSize = sampler_x.sample(maxSize_m)
    # Координаты датчиков для регистрации поля в м
    probesPos_m = [maxSize_m * 2 / 3]
    # Датчики для регистрации поля
    probesPos = [sampler_x.sample(pos) for pos in probesPos_m]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]
    # Положение источника в отсчетах
    sourcePos = sampler_x.sample(sourcePos_m)
    # Параметры гармонического сигнала
    lambda_wave = constants.c / frequency
    # Количество ячеек на длину волны
    Nl = (lambda_wave / dx)


    # Параметры среды
    # Диэлектрическая проницаемость
    eps = np.ones(maxSize)
    eps[:] = 4.0
    # Магнитная проницаемость
    mu = np.ones(maxSize - 1)
    
    Ez = np.zeros(maxSize)      
    Hy = np.zeros(maxSize - 1)

    # Источник
    source = GaussianModPlaneWave(30, 12, Nl, Sc, eps[sourcePos], mu[sourcePos])

    # Коэффициенты для расчета ABC второй степени
    # Sc' для левой границы
    Sc1Left = Sc / np.sqrt(mu[0] * eps[0])

    k1Left = -1 / (1 / Sc1Left + 2 + Sc1Left)
    k2Left = 1 / Sc1Left - 2 + Sc1Left
    k3Left = 2 * (Sc1Left - 1 / Sc1Left)
    k4Left = 4 * (1 / Sc1Left + Sc1Left)

    # Ez[0: 2] в предыдущий момент времени (q)
    oldEzLeft1 = np.zeros(3)
    # Ez[0: 2] в пред-предыдущий момент времени (q - 1)
    oldEzLeft2 = np.zeros(3)

    # Расчет коэффициентов для граничных условий ABC первой степени справа
    tempRight = Sc / np.sqrt(mu[-1] * eps[-1])
    koeffABCRight = (tempRight - 1) / (tempRight + 1)
    # Ez[-2] в предыдущий момент времени
    oldEzRight = Ez[-2]

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, B/m'
    display_ymin = -2.5
    display_ymax = 2.5

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(dx, dt, maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])

    for q in range(maxTime):
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= (Sc / (W0 * mu[sourcePos])) * source.getE(0, q)

        # Расчет компоненты поля E
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy[:-1]) * Sc * W0 / eps[1:-1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / np.sqrt(eps[sourcePos] * mu[sourcePos])) * source.getE(-0.5, q + 0.5)
        
        # Граничные условия ABC второй степени (слева)
        Ez[0] = (k1Left * (k2Left * (Ez[2] + oldEzLeft2[0]) +
                           k3Left * (oldEzLeft1[0] + oldEzLeft1[2] - Ez[1] - oldEzLeft2[1]) -
                           k4Left * oldEzLeft1[1]) - oldEzLeft2[2])

        oldEzLeft2[:] = oldEzLeft1[:]
        oldEzLeft1[:] = Ez[0: 3]
        
        # граничное условие справа
        Ez[-1] = oldEzRight + koeffABCRight * (Ez[-2] - Ez[-1])
        oldEzRight = Ez[-2]

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 5 == 0:
            display.updateData(display_field, q)

    display.stop()
    tools.showResult(probes, dx, dt, -5, 5)
    
