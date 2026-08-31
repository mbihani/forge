# Forge Orchestrator deployment catalog

Every deployment setting is declared once as a bundle variable in
`databricks.yml`. Search for its `CONFIGURE` marker to find the source of
truth. Keep this catalog and those markers in sync.

## Required

- `CONFIGURE(omnigent_server_url)` — URL of the Omnigent control-plane server.
- `CONFIGURE(omnigent_auth_token)` — bearer token accepted by that server.
- `CONFIGURE(git_remote_url)` — Git remote containing the scaffold to optimize.
- `CONFIGURE(ai_gateway_url)` — Databricks AI Gateway Anthropic endpoint.

## Customize

- `CONFIGURE(eval_engine)` — evaluation backend; defaults to `mlflow`.
- `CONFIGURE(mlflow_experiment)` — experiment path for run and trace data.
- `CONFIGURE(optimizer_model)` — optimizer model; defaults to Claude Opus 4.7.
- `CONFIGURE(domain_config)` — repository-relative domain configuration path.

## Optional

- `CONFIGURE(git_token)` — credential for a private Git remote; omit for public
  repositories or when workload identity supplies access.
- `CONFIGURE(databricks_workspace_id)` — numeric Databricks workspace/org ID used
  for the `?o=` param on Omnigent session UI links; omit to drop the param (the
  link resolves without it).
