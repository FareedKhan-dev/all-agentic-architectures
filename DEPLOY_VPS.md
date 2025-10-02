
## Environnements (templates)

- Script recommandé:

```bash
bash scripts/copy_env_templates.sh prod
# => copie backend/ops/env_templates/.env.production.example -> /etc/all-agentic/.env
```

- Copie manuelle:
```bash
sudo mkdir -p /etc/all-agentic
sudo cp backend/ops/env_templates/.env.production.example /etc/all-agentic/.env
sudo chmod 600 /etc/all-agentic/.env
```
