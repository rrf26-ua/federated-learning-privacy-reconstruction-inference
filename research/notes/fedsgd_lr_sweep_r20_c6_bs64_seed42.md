# FedSGD learning-rate sweep — CIFAR-10, ResNet-18

## Objetivo

Comparar diferentes tasas de aprendizaje para el modo FedSGD implementado en Flower y seleccionar una configuración razonable como baseline.

## Configuración común

- Dataset: CIFAR-10
- Modelo: ResNet-18 adaptada a CIFAR-10
- Algoritmo: FedSGD
- Clientes: 6
- Rondas: 20
- Batch size: 64
- Seed: 42
- Participación: 100%
- Evaluación: centralizada en servidor sobre test CIFAR-10

## Resultados

| Learning rate | Accuracy final | Mejor accuracy | Loss final | Interpretación |
|---:|---:|---:|---:|---|
| 0.02 | 0.2065 | 0.2065 | 2.1497 | Estable pero lento |
| 0.05 | 0.2473 | 0.2473 | 2.0385 | Mejor equilibrio |
| 0.1 | 0.2275 | 0.2559 | 2.3772 | Más inestable |

## Interpretación

FedSGD converge de forma mucho más lenta que FedAvg bajo el mismo número de rondas federadas. Esto es esperable, ya que FedAvg realiza varias épocas locales por ronda, mientras que FedSGD solo aplica un update global efectivo por ronda.

La configuración con learning rate 0.02 es estable, pero demasiado lenta. La configuración con learning rate 0.1 alcanza un pico de accuracy mayor, pero muestra oscilaciones fuertes en la loss y en la accuracy. La configuración con learning rate 0.05 ofrece el mejor equilibrio entre estabilidad y rendimiento final.

## Decisión experimental

Se selecciona `learning-rate=0.05` como baseline principal de FedSGD para los experimentos posteriores.

## Limitación

La comparación con FedAvg no representa igualdad de cómputo, sino igualdad de rondas federadas. Por tanto, debe interpretarse como una comparación bajo el mismo presupuesto de comunicación, no bajo el mismo número de pasos de optimización.
