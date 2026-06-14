# FedSGD baseline — CIFAR-10, ResNet-18

## Objetivo

Validar la implementación del modo FedSGD y obtener una primera línea base de rendimiento para compararla con FedAvg.

## Configuración

- Dataset: CIFAR-10
- Modelo: ResNet-18 adaptada a CIFAR-10
- Algoritmo: FedSGD implementado como un único update basado en el gradiente medio local de cada cliente
- Clientes: 6
- Rondas: 5
- Batch size del dataloader: 64
- Learning rate: 0.02
- Seed: 42
- Participación: 100%
- Evaluación: centralizada en el servidor sobre test CIFAR-10

## Resultados

| Ronda | Train loss | Test loss | Test accuracy |
|---:|---:|---:|---:|
| 0 | - | 2.9329 | 0.1000 |
| 1 | 2.3756 | 2.3163 | 0.0930 |
| 2 | 2.3111 | 2.2969 | 0.1062 |
| 3 | 2.2920 | 2.2837 | 0.1236 |
| 4 | 2.2796 | 2.2727 | 0.1367 |
| 5 | 2.2693 | 2.2629 | 0.1471 |

## Interpretación

El experimento valida que el modo FedSGD funciona correctamente: los 6 clientes completan el entrenamiento en cada ronda y el servidor recibe 6 resultados y 0 fallos.

La accuracy final es baja, pero esto es esperable. Con 5 rondas, FedSGD realiza solo 5 actualizaciones globales efectivas, mientras que la baseline FedAvg fuerte usa varias épocas locales por ronda. Por tanto, esta ejecución sirve como smoke test extendido y primera línea base, pero no como comparación justa de rendimiento final frente a FedAvg.

## Conclusión

FedSGD queda implementado y validado. El siguiente paso es ejecutar una configuración con más rondas para obtener una curva de convergencia más representativa.
