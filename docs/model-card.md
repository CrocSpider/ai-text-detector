# Model Card

## Version

`heuristic-ensemble-v0`

## Intended use

Advisory risk estimation for machine-generated or heavily machine-edited text.

## Core signals

- classifier placeholder based on stylometric and structural features
- stylometric anomaly scoring
- surprisal proxy based on repetition and entropy
- cross-segment consistency analysis
- extraction quality and language support penalties

## Known limitations

- not reliable for very short text
- performance is weaker for unsupported languages
- heavily edited or mixed-origin documents may be inconclusive
