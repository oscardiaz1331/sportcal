# Experiment write-ups

One file per sport (`hockey.md`, `soccer.md`, `tennis.md`, `basketball.md`). Every experiment gets
a section in this shape so results stay comparable and dead ends stay dead:

```markdown
## N. <Short name> - `lab/<sport>/<module>.py`

**Question:** what were we trying to find out?
**Method:** the smallest description that lets someone reproduce it (command, dataset, val set).
**Result:** numbers, in a table, on the reference validation set. State n and the sampling noise.
**Decision:** adopted / discarded / needs more data - and what would change it.
**Caveats:** known biases (circular val, correlated frames, transductive leakage).
```

Rules:

* Report **numbers from the reference validation set** (for hockey: `hockeyrink_nhl_valh`) and say which val set was used.
* A negative result is a result: record the exact figure so nobody retries it.
* Do not put results in code comments. A code docstring may link to a section here.
* Decisions that change the architecture or the product get an ADR in `docs/decisions/`, linked from here.
