# Troubleshooting

This page collects the most common runtime warnings and errors along with quick fixes. Copy the exact commands to resolve issues inside the container (or adapt them to your environment).

## `KeyError: 'default'` when calling `/v1/audio/speech`

**Symptom**

```
ERROR - Error loading voice: default, KeyError: 'default'
```

**Fix**

The request is using `voice="default"`, but the selected model namespace (for example `tts-1-hd`) does not define a `default:` mapping inside `config/voice_to_speaker.yaml`. Add a `default` entry that points to one of the configured voices, and make sure the YAML file is saved as UTF-8 without a BOM. After updating the file you can retry without restarting the container.

```
tts-1-hd:
  default: alloy
  alloy:
    model: xtts
    speaker: voices/alloy.wav
```

## `Unable to find voice: <name>` or “download voices” audio playback

**Symptom**

```
ERROR - Voice 'nova' model missing: voices/en_US-libritts_r-medium.onnx
```

**Fix**

The ONNX voice model file is not present under `/app/voices`. Download it inside the container:

```bash
docker exec -it <container> python3 -c "from pathlib import Path; import piper.download_voices as dl; dl.download_voice('<voice_code>', Path('/app/voices'))"
```

Replace `<voice_code>` with the identifier that matches the missing voice (for example `en_US-libritts_r-medium`).

## `/sys/class/drm/card0` warnings during startup

**Symptom**

```
[W:onnxruntime:, provider_bridge_ort.cc:1793 GetDefaultBackends] /sys/class/drm/card0 does not exist
```

**Fix**

This warning is harmless and ONNX Runtime falls back to the available device. Set `ORT_LOG_SEVERITY_LEVEL=3` in the environment (already included in the compose example) to silence the message.

## `pthread_setaffinity_np failed ... Invalid argument`

**Symptom**

```
WARNING - pthread_setaffinity_np failed with error 22
```

**Fix**

The container’s CPU affinity does not match the cpuset exposed by the host (common with LXC). Set the thread-related environment variables to `1` (see the docker-compose example) or adjust the host cpuset to include the allocated CPUs.

## `dockerfile parse error ... APT-GET`

**Symptom**

```
dockerfile parse error: Unknown instruction: APT-GET
```

**Fix**

Older Docker builders do not support heredoc syntax. The refreshed Dockerfile avoids heredocs entirely; rebuild with the updated file or enable BuildKit: `DOCKER_BUILDKIT=1 docker build ...`.
