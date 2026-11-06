"""
Shadow-mode decision model.

A candidate decision model (a pinned git commit in its own worktree) runs next
to the live engine on production data without being able to affect it:

* each cycle snapshots the production DB with SQLite's online backup API;
* the candidate runs in a subprocess against that *copy* only;
* results go to a separate SQLite file (``data/db/shadow/shadow_decisions.db``)
  together with the live engine's decision for the same moment.

Nothing in this package writes to the production database, sends Telegram
messages, or places trades.
"""
