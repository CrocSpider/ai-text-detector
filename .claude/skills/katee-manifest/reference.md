# Katee Manifest Reference

## How this was derived

The current Katee manifests were not copied from GCP APIs. They were derived by inspecting live Katee/KubeVela resources with `kubectl`, then matching the manifest shape to what the cluster actually accepted and rendered.

The key signals were:

1. rendered `Application`, `Job`, and `Pod` behavior
2. failures from the first raw Kubernetes manifest attempt
3. server-side dry-run validation against the cluster
4. scheduling and runtime events from the real pods

## Current accepted shape

- `apiVersion: core.oam.dev/v1beta1`
- `kind: Application`
- component type `snjob`
- trait type `snstorage`

## Important Katee behavior

- Prefer Katee-native `Application` manifests over raw `Job` manifests.
- The platform may transform generic Kubernetes manifests in ways that drop mounts or alter runtime shape.
- For `snjob`, CPU and memory belong under:

```yaml
properties:
  resources:
    cpu: "4"
    memory: 24Gi
```

- For storage and mounted config, use `snstorage` sections such as `configMap`, `pvc`, and `emptyDir`.
- To reuse an existing PVC, use `mountOnly: true`.

## Current GPU default

Use this unless cluster evidence suggests otherwise:

```yaml
properties:
  computeClassEnabled: true
  computeClass: nvidia-l4-1
```

This default was chosen after observing cluster scheduling behavior and avoiding unavailable or more expensive classes.

## Current training files

- HC3 prep app: `infra/k8s/training/katee-application.prepare-hc3.example.yaml`
- Main training app: `infra/k8s/training/katee-application.gpu.from-existing-data.example.yaml`
- Training notes: `docs/training-on-kubernetes.md`

## Current image baseline

- `REGISTRY/PROJECT/TEAM/ai-text-detector-trainer:1.3`

That tag includes the `transformers` compatibility fix for the `TrainingArguments` API change.

## Data flow for training

1. Run the HC3 prep application.
2. It creates or fills `ai-text-detector-data` with:
   - `train.jsonl`
   - `validation.jsonl`
   - `test.jsonl`
3. Run the training application.
4. The training app reads local HC3 files from the PVC and pulls `TuringBench-tokenized` and `RAID` from Hugging Face at runtime.

## Validation commands

```bash
kubectl apply --dry-run=server -f infra/k8s/training/katee-application.prepare-hc3.example.yaml
kubectl apply --dry-run=server -f infra/k8s/training/katee-application.gpu.from-existing-data.example.yaml
```

## Runtime checks

```bash
kubectl config current-context
kubectl config view --minify --output 'jsonpath={..namespace}'
kubectl get application ai-text-detector-train -o wide
kubectl get jobs -o wide | grep ai-text-detector
kubectl get pods -o wide | grep ai-text-detector
kubectl logs job/<job-name> -f
kubectl describe job <job-name>
kubectl describe pod <pod-name>
```

## Known gotchas

- A healthy `Application` does not guarantee the underlying `Job` succeeded.
- `RAID` is large; use sliced `hf://...split=train[:N]` URIs for practical first runs.
- The first visible activity can be CPU-heavy preprocessing before any GPU utilization appears.
- PVC mount problems, computeClass mismatches, and library API changes are the first things to check when a run fails.
