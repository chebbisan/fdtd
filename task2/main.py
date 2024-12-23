# -*- coding: utf-8 -*-

import math

import numpy as np
import numpy.typing as npt

from objects import LayerContinuous, LayerDiscrete, Probe

import boundary
import sources
import tools


class Sampler:
    def __init__(self, discrete: float):
        self.discrete = discrete

    def sample(self, x: float) -> int:
        return math.floor(x / self.discrete + 0.5)


def sampleLayer(layer_cont: LayerContinuous, sampler: Sampler) -> LayerDiscrete:
    start_discrete = sampler.sample(layer_cont.xmin)
    end_discrete = (sampler.sample(layer_cont.xmax)
                    if layer_cont.xmax is not None
                    else None)
    return LayerDiscrete(start_discrete, end_discrete,
                         layer_cont.eps, layer_cont.mu, layer_cont.sigma)


def fillMedium(layer: LayerDiscrete,
               eps: npt.NDArray[np.float64],
               mu: npt.NDArray[np.float64],
               sigma: npt.NDArray[np.float64]):
    if layer.xmax is not None:
        eps[layer.xmin: layer.xmax] = layer.eps
        mu[layer.xmin: layer.xmax] = layer.mu
        sigma[layer.xmin: layer.xmax] = layer.sigma
    else:
        eps[layer.xmin:] = layer.eps
        mu[layer.xmin:] = layer.mu
        sigma[layer.xmin:] = layer.sigma


if __name__ == '__main__':

    # Используемые константы
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * np.pi

    # Скорость света в вакууме
    c = 299792458.0

    # Электрическая постоянная
    eps0 = 8.854187817e-12

    # Параметры моделирования
    # Частота сигнала, Гц
    f_Hz = 1e9

    # Дискрет по пространству в м
    # for test Hz 0.2*1e-4
    # for model 2*1e-4
    dx = 0.5*1e-4
    
    wavelength = c / f_Hz
    Nl = wavelength / dx

    # Число Куранта
    Sc = 1.0

    # Размер области моделирования в м
    maxSize_m = 1

    # Время расчета в секундах
    maxTime_s = 9e-9

    # Положение источника в м
    sourcePos_m = 0.3

    # Координаты датчиков для регистрации поля в м
    probesPos_m = [0.125, 0.3]

    # Параметры слоев
    layers_cont = [LayerContinuous(0.5, 0.56,  eps=3.5, sigma=0.0),
                   LayerContinuous(0.56, 0.62 , eps=2.2, sigma=0.0),
                   LayerContinuous(0.62, 0.72, eps=2, sigma=0.0),
                   LayerContinuous(0.72, eps=6, sigma=0.0)
                   ]

    # Скорость обновления графика поля
    speed_refresh = 30

    # Переход к дискретным отсчетам
    # Дискрет по времени
    dt = dx * Sc / c

    sampler_x = Sampler(dx)
    sampler_t = Sampler(dt)

    # Время расчета в отсчетах
    maxTime = sampler_t.sample(maxTime_s)

    # Размер области моделирования в отсчетах
    maxSize = sampler_x.sample(maxSize_m)

    # Положение источника в отсчетах
    sourcePos = sampler_x.sample(sourcePos_m)

    layers = [sampleLayer(layer, sampler_x) for layer in layers_cont]

    # Датчики для регистрации поля
    probesPos = [sampler_x.sample(pos) for pos in probesPos_m]
    probes = [Probe(pos, maxTime) for pos in probesPos]

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = np.ones(maxSize)

    # Магнитная проницаемость
    mu = np.ones(maxSize - 1)

    # Проводимость
    sigma = np.zeros(maxSize)

    for layer in layers:
        fillMedium(layer, eps, mu, sigma)

    # Коэффициенты для учета потерь
    loss = sigma * dt / (2 * eps * eps0)
    ceze = (1.0 - loss) / (1.0 + loss)
    cezh = W0 / (eps * (1.0 + loss))


    # Источник
    magnitude = 1.0
    signal = sources.Gaussian(magnitude, 4000, 800)
    source = sources.SourceTFSFG(signal, 0.0, Sc, eps[sourcePos], mu[sourcePos])

    Ez = np.zeros(maxSize)
    Hy = np.zeros(maxSize - 1)

    # Создание экземпляров классов граничных условий
    boundary_left = boundary.ABCSecondLeft(eps[0], mu[0], Sc)
    boundary_right = boundary.ABCSecondRight(eps[-1], mu[-1], Sc)

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -2.1
    display_ymax = 2.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(dx, dt,
                                        maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel,
                                        title='fdtd_dielectric')

    display.activate()
    display.drawSources([sourcePos])
    display.drawProbes(probesPos)
    for layer in layers:
        display.drawBoundary(layer.xmin)
        if layer.xmax is not None:
            display.drawBoundary(layer.xmax)

    for t in range(1, maxTime):
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] += source.getH(t)

        # Расчет компоненты поля E
        Ez[1:-1] = ceze[1: -1] * Ez[1: -1] + cezh[1: -1] * (Hy[1:] - Hy[: -1])

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += source.getE(t)

        boundary_left.updateField(Ez, Hy)
        boundary_right.updateField(Ez, Hy)

        for i, probe in enumerate(probes):
            if i == 1 and t >= 7500:
                continue
            probe.addData(Ez, Hy)

        if t % speed_refresh == 0:
            display.updateData(display_field, t)

    display.stop()

    # Отображение сигнала, сохраненного в пробнике
    tools.showProbeSignals(probes, dx, dt, -2.1, 2.1)
    tools.showProbeSpectre(probes, dx, dt, -2.1, 2.1)
    tools.showGamma(probes, dx, dt, -2.1, 2.1)
