# Install PhysicsVQA Frontend locally Using npm
# Physics Question Answering (PhysicsVQA)

Comprehensive platform for physics question answering — combines a web frontend, a Node.js API backend, and an optional model-serving Python backend (vLLM/VLLM-like). This repository provides tools for running locally (recommended) or with Docker Compose for a quick setup.

Key components:
- `client/` — frontend (Vite/React) for user interactions.
- `api/` — backend server (Node.js) that handles requests, routing, and retrieval.
- `vllm_backend/` — optional Python model server for serving large language models (requires GPU & vLLM-compatible setup).
- `docker-compose.yml` — compose stack to run the full system in containers.
- `config/` — utility scripts and operations for deployment and maintenance.

## Features
- Interactive web UI for asking physics questions and receiving step-by-step answers.
- Extensible architecture to plug in different model backends (local model server, remote LLM API).
- Caching and rate-limiting helpers for production readiness.

## Quick Start (recommended): Docker Compose
This will build and start all components (frontend, backend, optional model server) with a single command. Docker Desktop is required on Windows.

Open a PowerShell terminal and run:

```powershell
docker-compose up --build
```

After the services start, open the frontend in your browser at:

```
http://localhost:3080/
```

Notes:
- If you only want the web UI and API (without the Python model server), edit `docker-compose.yml` and remove/disable the `vllm_backend` service, or run the specific services: `docker-compose up --build api client`.
- On first run Docker will download images and build containers — this may take a while.

## Local Development (manual)
Use this when you want to actively develop frontend or backend code without containers.

Prerequisites:
- Node.js (v16+ recommended) and npm or pnpm
- Python 3.10+ (if running `vllm_backend` locally)

1) Frontend

```powershell
cd client
npm install        # or `pnpm install`
npm run dev        # starts Vite dev server (check package.json scripts)
```

Default dev URL is usually `http://localhost:5173/` (Vite). See the terminal output for exact URL.

2) API (Node.js)

```powershell
cd api
npm install
npm run dev        # or `npm start` depending on scripts in api/package.json
```

3) Optional: Python model backend

```powershell
cd vllm_backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ../requirements.txt
python vllm_server.py   # or follow README inside vllm_backend for model-specific setup
```

Environment variables and configuration
- Configuration files and secrets are typically loaded from `config/` or environment variables.
- Copy any sample `.env.example` or follow `config/` scripts to set a local `.env`. Inspect `api/package.json` and `client/package.json` for env keys referenced by scripts.

## Running with a remote LLM provider
If you prefer not to run a local model server, set the API to use a remote LLM provider (OpenAI, Anthropic, etc.) by configuring the provider keys in the API environment and updating the provider setting in the API config. See `config/` for parsers and helper utilities.

## Tests
- JavaScript tests use Jest configurations found in `api/jest.config.js` and `client/jest.config.cjs`.

To run tests (example):

```powershell
cd api
npm test

cd ../client
npm test
```

Adjust to `pnpm` or `yarn` if you use those package managers.

## Troubleshooting
- If ports conflict, check `docker-compose.yml` for exposed ports and change them.
- For frontend build issues, delete `node_modules` and run a fresh `npm ci`/`npm install`.
- If the Python model server fails to start, confirm CUDA drivers (for GPU) and the exact model/runtime requirements in `vllm_backend/README` (if applicable).

## Contributing
- Fork the repo and open a branch for your feature or bugfix.
- Keep changes small and focused and open a pull request describing the change.
- Run frontend and backend tests locally before submitting a PR.

## Project Structure (short)
- `client/` — frontend app (Vite, React/TS/JS depending on the folder contents)
- `api/` — Node.js backend and route handlers
- `vllm_backend/` — optional Python model server
- `config/` — scripts and utilities for deployment and maintenance
- `docker-compose.yml` — compose file for local full-stack testing

## Contact & Credits
This project is part of the Fuse AI Fellowship / community projects. For urgent issues open an issue in this repo and tag contributors.

## License
This repository contains a `LICENSE` file. Refer to it for license terms.

---
If you'd like, I can also:
- add a short quickstart script for a minimal local-only setup,
- create an example `.env.example` with common variables used by `api` and `client`, or
- run the project locally and report any runtime errors I encounter.

If you want one of those, tell me which and I'll continue.
