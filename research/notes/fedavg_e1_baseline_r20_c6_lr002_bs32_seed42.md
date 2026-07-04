# FedAvg E=1 baseline — CIFAR-10, ResNet-18

## Objetivo

Evaluar una configuración intermedia entre FedSGD estricto y FedAvg fuerte con múltiples épocas locales.

## Configuración

- Dataset: CIFAR-10
- Modelo: ResNet-18 adaptada a CIFAR-10
- Algoritmo cliente: FedAvg
- Estrategia servidor: FedAvg
- Clientes: 6
- Rondas: 20
- Épocas locales: 1
- Learning rate: 0.02
- Batch size: 32
- Seed: 42
- Participación: 100%
- Evaluación: centralizada en servidor sobre test CIFAR-10

## Resultados principales

- Accuracy inicial: 0.1000
- Accuracy final: 0.7501
- Mejor accuracy: 0.7501
- Loss final: 0.7824
- Tiempo total: 438.52 s

## Interpretación

FedAvg con una única época local por ronda alcanza un rendimiento claramente superior al FedSGD estricto. Esto confirma que la baja accuracy de FedSGD no se debe al modelo ni al framework, sino al número muy reducido de pasos efectivos de optimización.

Esta configuración representa una baseline intermedia útil: mantiene un presupuesto de entrenamiento más moderado que FedAvg con 10 épocas locales, pero ya proporciona un rendimiento suficientemente alto para experimentos posteriores de privacidad y defensas.

## Comparación cualitativa

- FedSGD estricto: converge lentamente y alcanza aproximadamente 31% de accuracy tras 50 rondas.
- FedAvg E=1: alcanza 75.01% tras 20 rondas.
- FedAvg E=10: alcanza 86.72% tras 5 rondas en la baseline fuerte previa.

## Decisión experimental

Se conserva FedSGD estricto como escenario de contraste y exposición tipo gradiente. Para experimentos posteriores, FedAvg E=1 puede usarse como baseline intermedia y FedAvg E=10 como baseline fuerte de rendimiento.
