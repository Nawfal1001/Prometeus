# ============================================================
#  PROMETHEUS — Multi-bot subsystem
#
#  Each "bot" is an independent trading engine running in its
#  own OS subprocess with a fully isolated config (own symbols,
#  exchange, mode, ML model, optimization params and creds).
#
#  Why subprocess: config.settings is a global singleton, so two
#  engines in one process share it. One process per bot gives
#  clean isolation + independent crash/restart, exactly like the
#  optimizer's out-of-process runner.
# ============================================================
