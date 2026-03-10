# Training On Kubernetes

The repository includes a GPU-oriented training image in `services/ml` and example Kubernetes manifests in `infra/k8s/training`.

If you are deploying through Katee/KubeVela, use `infra/k8s/training/katee-application.gpu.example.yaml` instead of the raw Job manifest.

## Recommended workflow

1. Prepare `train.jsonl`, `validation.jsonl`, and `test.jsonl` on a mounted volume.
2. Update `infra/k8s/training/all-in-one.gpu.example.yaml` with the right paths, PVC names, and model settings.
3. Build and push the training image from `services/ml/Dockerfile`.
4. Apply the single manifest file.
5. After training completes, point the API at the produced artifact directory.

## Single-file apply

```bash
kubectl apply -f infra/k8s/training/all-in-one.gpu.example.yaml
```

The example job in that file already points to:

```text
REGISTRY/PROJECT/TEAM/ai-text-detector-trainer:1.3
```

## Katee apply

```bash
kubectl apply -f infra/k8s/training/katee-application.prepare-hc3.example.yaml
kubectl apply -f infra/k8s/training/katee-application.gpu.from-existing-data.example.yaml
```

The Katee manifests use the native `Application` + `snjob` + `snstorage` path so PVCs and config mounts are preserved correctly.

## Where the files come from

The training job expects three JSONL files in the data volume:

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`

The quickest way to create them is to run the HC3 preparation job. It downloads the public `Hello-SimpleAI/HC3` dataset and writes those files into the `ai-text-detector-data` PVC.

You do not need to manually download `TuringBench-tokenized` or `RAID` if your cluster can reach Hugging Face. The mixed-source training config pulls those automatically at runtime via `hf://...` dataset URIs.

For very large datasets like `RAID`, use split slicing in the URI, for example:

```text
hf://liamdugan/raid?subset=raid&split=train[:50000]
```

That keeps the first pass manageable without having to manually stage multi-million-row files.

## Minimum dataset row

```json
{"document_id":"doc-123","text":"...","label":0}
```

## Strongly recommended metadata

- `language`
- `domain`
- `writer_profile`
- `attack_type`
- `source_dataset`
- `extraction_quality`

## Output

The training run emits a `manifest.json` plus model artifacts and evaluation reports. The API can load the run directly if the optional ML dependencies are installed.
