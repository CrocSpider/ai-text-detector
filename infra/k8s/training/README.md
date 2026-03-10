# Training Manifests

This directory contains example Kubernetes manifests for GPU training.

- `katee-application.gpu.example.yaml`: Katee-native single-file `Application` manifest for the training job.
- `katee-application.prepare-hc3.example.yaml`: Katee-native job that downloads HC3 and writes `train.jsonl`, `validation.jsonl`, and `test.jsonl` into the data PVC.
- `katee-application.gpu.from-existing-data.example.yaml`: Katee-native training job that expects an existing `ai-text-detector-data` PVC.
- `katee-application.gpu.a100.from-existing-data.example.yaml`: Katee-native A100-oriented training job that expects an existing `ai-text-detector-data` PVC.
- `all-in-one.gpu.example.yaml`: one-file manifest with PVCs, config map, and GPU job.
- `pvc.example.yaml`: persistent volumes for datasets, artifacts, and cache.
- `configmap.example.yaml`: example training config mounted at `/config/train.yaml`.
- `job.gpu.example.yaml`: example batch job that runs `python -m trainer.cli train --config /config/train.yaml`.

The quickest path is:

```bash
kubectl apply -f infra/k8s/training/all-in-one.gpu.example.yaml
```

For Katee, use:

```bash
kubectl apply -f infra/k8s/training/katee-application.prepare-hc3.example.yaml
kubectl apply -f infra/k8s/training/katee-application.gpu.from-existing-data.example.yaml
```

For an A100-class GPU run, use:

```bash
kubectl apply -f infra/k8s/training/katee-application.gpu.a100.from-existing-data.example.yaml
```

That manifest updates the existing `ai-text-detector-train` application in place. Katee does not allow a second application to take ownership of the same config map and PVC-backed resources.

Update image names, PVC names, and config values before applying them.
