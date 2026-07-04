# Instrucciones para Codex en este repositorio

## Contexto del proyecto

Este repositorio corresponde a un Trabajo Fin de Grado sobre privacidad y seguridad en aprendizaje federado. El objetivo actual es estudiar si la baseline FedSGD puede mejorar su accuracy de forma rigurosa, sin alterar los resultados principales ya obtenidos con FedAvg, ataques de reconstrucción, clipping y ruido gaussiano.

## Reglas estrictas

- No modificar los scripts de ataques salvo que sea necesario para leer resultados existentes.
- No modificar las defensas ya implementadas.
- No borrar ni sobrescribir resultados existentes.
- No cambiar CIFAR-10, ResNet-18, normalización, partición de clientes ni evaluación centralizada sin justificarlo explícitamente.
- No cambiar FedSGD para que deje de ser FedSGD. No convertirlo en FedAvg con varias épocas locales.
- No perseguir solo una métrica: guardar todos los intentos, incluidos los malos.
- No hacer commits automáticos sin mostrar antes el diff.
- No añadir dependencias pesadas salvo que sean imprescindibles.
- No subir data/, .venv/ ni checkpoints .pt/.pth.

## Objetivo técnico actual

Crear una rama experimental para mejorar la configuración FedSGD mediante un barrido controlado de hiperparámetros. El objetivo práctico es comprobar si FedSGD puede alcanzar al menos 60-65% de accuracy en CIFAR-10 con ResNet-18. Si no se alcanza, documentar la mejor configuración y explicar por qué FedSGD no se utilizará como baseline principal.

## Resultados esperados

Guardar una tabla CSV en:

results/experiment_summaries/fedsgd_sweep_summary.csv

con columnas:
config_id, rounds, clients, batch_size, lr, weight_decay, scheduler, final_loss, final_accuracy, best_accuracy, elapsed_time, command.

## Criterio de parada

Detener el barrido si una configuración supera 65% de accuracy o si se han probado todas las configuraciones razonables definidas en el prompt principal.
