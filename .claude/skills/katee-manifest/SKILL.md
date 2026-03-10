---
name: katee-manifest
description: Create, validate, or debug Springernature Katee/KubeVela Application manifests. Use when a workload must run on Katee with snjob and snstorage, especially for GPU jobs, PVC mounts, config maps, or computeClass-specific scheduling.
argument-hint: "[goal, manifest path, or workload description]"
allowed-tools: Read, Edit, Write, Glob, Grep, Bash(kubectl *), Bash(helm *), Bash(gcloud *), Bash(docker *), Bash(python *), Bash(ruby *)
---

Use this skill when working on Katee manifests.

If `$ARGUMENTS` is present, treat it as the requested workload, manifest path, or debugging target.

## Workflow

1. Inspect existing repo manifests under `infra/` and any current Katee examples before editing.
2. If cluster access is available, inspect the live resources to confirm what Katee actually renders:
   - `kubectl get application ...`
   - `kubectl get jobs -o wide`
   - `kubectl get pods -o wide`
   - `kubectl describe job ...`
   - `kubectl describe pod ...`
3. Prefer a Katee-native `Application` manifest over a raw Kubernetes `Job` unless the user explicitly asks for generic Kubernetes.
4. Use the current accepted shape:
   - `apiVersion: core.oam.dev/v1beta1`
   - `kind: Application`
   - component type `snjob`
   - storage trait `snstorage`
5. Put command, image, env, retry behavior, and TTL under `spec.components[].properties`.
6. Put CPU and memory under `properties.resources.cpu` and `properties.resources.memory`.
7. For storage, use `traits: - type: snstorage` and choose the right section:
   - `configMap` for inline config files mounted into the container
   - `pvc` for persistent volumes
   - `emptyDir` for scratch/cache space
8. If reusing an existing PVC, set `mountOnly: true`.
9. For GPU workloads, prefer `computeClassEnabled: true` and an explicit `computeClass`. Use `nvidia-l4-1` as the current default unless the cluster clearly supports a better class.
10. Validate every manifest with `kubectl apply --dry-run=server -f <manifest>` before treating it as correct.
11. After apply, verify the rendered `Job` and `Pod`. Do not assume a healthy `Application` means the actual training run succeeded.
12. If a run fails, inspect scheduling events, PVC attachment, image start, and runtime stack traces before changing the manifest.

## Manifest rules

- Start from the generic template in [templates/snjob-application.yaml](templates/snjob-application.yaml) and adapt it to the workload.
- Reuse the current repo examples in `infra/k8s/training/` when the requested workload is related to training or dataset prep.
- Keep namespace, labels, and Katee annotations explicit.
- Preserve PVC reuse vs PVC creation semantics carefully.
- If the manifest is for a training workflow, keep data, artifact, and cache mounts separate.

## Debugging rules

- When a raw Kubernetes manifest behaves strangely on Katee, assume the platform may be transforming it.
- If the pod is pending, inspect `kubectl describe pod` for computeClass, GPU, and storage events.
- If the pod starts but the workload fails, inspect logs for runtime or library API mismatches.
- If the user says the job "completed" or the app is "healthy", still verify the underlying `Job` exit code and pod logs.

## Output expectations

When you create or update a Katee manifest, explain:

- what manifest file you changed or created
- what assumptions you made about namespace, image, computeClass, and storage
- how to validate it with `kubectl`
- any cluster-specific risks that still need verification

## Additional resources

- Detailed background and current cluster-specific guidance: [reference.md](reference.md)
- Generic starting template: [templates/snjob-application.yaml](templates/snjob-application.yaml)
