# Katee Manifest Handoff

## Short answer

No, the Katee manifest was not pulled from GCP directly.

It was derived by inspecting the live Katee/KubeVela resources in the cluster with `kubectl`, then matching the manifest to the schema and behavior the cluster actually accepted.

## How the manifest was derived

The working Katee manifests were built from a few signals:

1. **Live cluster resources**
   - inspected the rendered `Application`, `Job`, and `Pod`
   - checked what Katee actually produced after applying manifests

2. **Behavior of the first failed attempt**
   - the raw Kubernetes job path got transformed by the platform
   - that rendered job dropped mounts/resources we needed
   - that told us to stop using the raw manifest path for this environment

3. **Katee/OAM shape accepted by the cluster**
   - `apiVersion: core.oam.dev/v1beta1`
   - `kind: Application`
   - component type `snjob`
   - storage trait `snstorage`

4. **Server-side validation**
   - validated with `kubectl apply --dry-run=server -f ...`

5. **Scheduling and runtime events**
   - pod events showed which GPU compute classes were available or unavailable
   - this is why the current training manifest uses `nvidia-l4-1`

## What was learned about Katee

- Prefer **Katee-native `Application` manifests** over raw Kubernetes jobs in this environment.
- For storage, use **`snstorage`**, not the raw PVC path from the first generic manifest.
- For `snjob`, CPU and memory belong under:

```yaml
properties:
  resources:
    cpu: "4"
    memory: 24Gi
```

- GPU scheduling is currently driven by:

```yaml
properties:
  computeClassEnabled: true
  computeClass: nvidia-l4-1
```

## Current files to use

- HC3 prep job: `infra/k8s/training/katee-application.prepare-hc3.example.yaml`
- Main training job: `infra/k8s/training/katee-application.gpu.from-existing-data.example.yaml`
- Training notes: `docs/training-on-kubernetes.md`

## Current expected image

- `REGISTRY/PROJECT/TEAM/ai-text-detector-trainer:1.4`

This tag includes the `transformers` compatibility fix for the `TrainingArguments` API mismatch.

## Current data flow

1. Run the HC3 prep application.
2. That creates or fills `ai-text-detector-data` with:
   - `train.jsonl`
   - `validation.jsonl`
   - `test.jsonl`
3. Run the training application.
4. The training app reads local HC3 files from the PVC and pulls `TuringBench-tokenized` and `RAID` from Hugging Face at runtime.

## Brief instructions for the next agent

1. Verify context and namespace:

```bash
kubectl config current-context
kubectl config view --minify --output 'jsonpath={..namespace}'
```

2. Check the current Katee application and rendered job:

```bash
kubectl get application ai-text-detector-train -o wide
kubectl get jobs -o wide | grep ai-text-detector
kubectl get pods -o wide | grep ai-text-detector
```

3. If the data PVC does not exist yet, run:

```bash
kubectl apply -f infra/k8s/training/katee-application.prepare-hc3.example.yaml
```

4. To run training, use:

```bash
kubectl apply -f infra/k8s/training/katee-application.gpu.from-existing-data.example.yaml
```

5. Follow progress with:

```bash
kubectl logs job/<job-name> -f
kubectl describe job <job-name>
kubectl describe pod <pod-name>
```

6. If training fails, check first for:
   - `transformers` API mismatches
   - unavailable compute class / GPU scheduling issues
   - PVC mount problems
   - Hugging Face download or cache issues

7. When the run completes, inspect the artifact directory and `manifest.json` before wiring the API to it.

## Known gotchas

- The platform may transform raw Kubernetes manifests, so do not assume a generic Job YAML will survive unchanged.
- `RAID` is large; use sliced `hf://...split=train[:N]` URIs for practical first runs.
- The first visible work may be CPU-heavy preprocessing and tokenization before any GPU utilization appears.
- A healthy `Application` does not automatically mean the underlying training `Job` succeeded.

## Recommended quick validation commands

```bash
kubectl apply --dry-run=server -f infra/k8s/training/katee-application.prepare-hc3.example.yaml
kubectl apply --dry-run=server -f infra/k8s/training/katee-application.gpu.from-existing-data.example.yaml
```
