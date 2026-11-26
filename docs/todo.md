# To do later

Small tasks that are not big enough for `docs/future_work.md` (which holds the larger, prioritised items).
Add a line when you find something, tick it (`[x]`) and note the result when it is done. Every number must be
measured; log behavioural changes in `docs/dev/changes_log.md`.

## Open

- [ ] **Compare the two prototype methods (`alpha_fps` vs `gcp`).**
  - *Why:* v2 keeps the in-house `alpha_fps` as default and implements the paper's algorithm (Al Muzaddid & Beksi,
    arXiv:2510.17043, Algorithm 1) as `gcp`, but no measurement has been made, so the default is not backed by
    evidence (`src/reid/identity/prototypes.py`, section 6.1 of `docs/v1_to_v2_report.md`).
  - *How:* same processed data, change only `--set association.proto_method=gcp` (prototypes are rebuilt at the
    end of a run, or by re-cluster in the app). Run `reid evaluate` on AI City scene 075 and compare held-out
    retrieval (Rank-1, Rank-5, mAP) for both. Try `gcp_alpha` at a few values too (0.5 is a default, not tuned).
  - *Done when:* a table with both methods in `docs/dev/changes_log.md`, and the default changed only if `gcp` is
    clearly better. Ideally repeat on the real warehouse showcase footage.

## Done

(nothing yet)
