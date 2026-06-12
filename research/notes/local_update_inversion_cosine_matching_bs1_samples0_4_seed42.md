# Local update inversion with cosine matching — CIFAR-10, ResNet-18

## Objetivo

Evaluar si la función de matching utilizada por el atacante afecta a la calidad de reconstrucción en ataques de inversión de updates locales.

Hasta ahora el ataque minimizaba una pérdida MSE/L2 entre el update local observado y el update simulado por la imagen dummy. En este experimento se sustituye esa pérdida por una pérdida basada en cosine similarity.

## Configuración

- Dataset: CIFAR-10
- Split: train
- Modelo: ResNet-18 adaptada a CIFAR-10
- Batch size: 1
- Samples: 0, 1, 2, 3, 4
- Grad/update scope: all
- Local LR: 0.02
- Weight decay: 0.0005
- Iteraciones: 2000
- Attack LR: 0.03
- TV weight: 1e-6
- Matching loss: cosine
- Seed: 42
- Etiquetas: se usa la etiqueta real para reconstrucción en este escenario controlado

## Resultados

| Sample | Label | Inferred label | Best-update PSNR | Best-oracle PSNR | Final PSNR | Tiempo |
|---:|---|---|---:|---:|---:|---:|
| 0 | airplane | airplane | 35.49 | 35.96 | 34.10 | 63.99 s |
| 1 | frog | frog | 17.24 | 17.68 | 17.68 | 65.15 s |
| 2 | airplane | airplane | 28.78 | 29.13 | 27.63 | 66.61 s |
| 3 | bird | bird | 29.76 | 29.99 | 29.98 | 62.29 s |
| 4 | horse | horse | 23.46 | 24.16 | 24.18 | 63.41 s |

La inferencia de etiqueta fue correcta en las cinco muestras.

## Comparación con MSE/L2

| Sample | Final PSNR con MSE | Final PSNR con cosine |
|---:|---:|---:|
| 0 | 17.40 | 34.10 |
| 1 | 9.57 | 17.68 |
| 2 | 12.19 | 27.63 |
| 3 | 10.74 | 29.98 |
| 4 | 10.91 | 24.18 |

La media del PSNR final pasa de aproximadamente 12.16 dB con MSE/L2 a aproximadamente 26.71 dB con cosine similarity.

## Interpretación

La elección de la función objetivo del atacante tiene un impacto crítico en la calidad de reconstrucción. La pérdida MSE/L2 sobre updates producía reconstrucciones parciales y muy variables, mientras que cosine similarity permite obtener reconstrucciones de alta fidelidad en varias muestras.

Esto sugiere que una evaluación basada únicamente en ataques con matching L2 puede infraestimar la vulnerabilidad real del sistema. En este escenario controlado, los updates locales de un paso contienen información suficiente para inferir correctamente la etiqueta y reconstruir imágenes privadas con alta fidelidad cuando se usa una función de matching más eficaz.

## Limitaciones

Este experimento sigue siendo un escenario controlado:

- Se usa batch size 1.
- Se usa la etiqueta real durante la reconstrucción para aislar la fuga visual.
- El update local se simula en el script, no se captura todavía desde una ronda Flower completa.
- Las conclusiones deben validarse posteriormente en configuraciones más realistas, como FedSGD/FedAvg comparativo, batches mayores, varios pasos locales y defensas.

## Conclusión

Cosine similarity mejora drásticamente la inversión de updates locales frente a la pérdida MSE/L2. Este resultado refuerza la vulnerabilidad de la baseline sin defensas y justifica avanzar hacia la comparación con FedSGD y la evaluación de defensas.
