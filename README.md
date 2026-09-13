![AI Generated](https://img.shields.io/badge/AI%20Generated-Gemini-blue?style=for-the-badge)

# OpenClaw-RL Local Proof-of-Concept

## 1. Information of the POC
This project is a fully containerized, local Reinforcement Learning (RL) Proof-of-Concept (POC) architecture designed to test the OpenClaw AI agent. It simulates an asynchronous RL loop by decoupling the environment, policy inference, process reward evaluation, and optimization into four isolated microservices. 

The architecture consists of:
*   **Agent Environment (`openclaw-gateway`):** The client interface and workspace sandbox running on port 18789.
*   **Policy Server (`policy_server`):** An Ollama instance serving the `qwen2.5:1.5b` model for local, CPU-friendly inference.
*   **Judge (`prm_server`):** A lightweight FastAPI Process Reward Model (PRM) that evaluates agent actions and assigns scalar rewards.
*   **Optimizer (`trainer_engine`):** A simulated PyTorch LoRA engine that performs asynchronous weight update loops on the CPU.


This is how it is supposed to work
- Agent to Inference: openclaw-gateway successfully points to policy_server over the native HTTP /api/chat route.
- Agent to Judge: openclaw-gateway correctly sends action payloads to the prm_server via the HTTP /evaluate endpoint.
- Optimizer to Judge: trainer_engine accurately shows the asynchronous polling mechanism used to fetch reward signals from the prm_server.


<img src="images/frist architecture image.png" style="width:650px;">

Project structure:

   ```text
   openclaw-rl/
   ├── .env
   ├── docker-compose.yml
   ├── README.md
   ├── mock_prm/
   │   ├── Dockerfile
   │   ├── main.py
   │   └── requirements.txt
   └── mock_trainer/
       ├── Dockerfile
       ├── train.py
       └── requirements.txt
   ```

## 2. Scope
**In-Scope:**
*   Demonstrating how OpenClaw connects to a local, offline LLM without exposing API keys to the internet.
*   Verifying tool-calling and workspace manipulation using OpenClaw's native Ollama integration.
*   Simulating an RL pipeline where agent trajectories are scored and passed to an asynchronous gradient optimizer.

**Out-of-Scope:**
*   Production deployment or exposure to the public internet (all ports are bound to `127.0.0.1`).
*   True GPU-accelerated training. The `trainer_engine` uses CPU wheels to simulate backpropagation mathematically without requiring NVIDIA hardware.
*   Handling massive LLMs. The context is constrained, and inference is limited to the 1.5B parameter model to prevent CPU/RAM crashes.

## 3. How to Build

### Prerequisites
*   Windows Subsystem for Linux (WSL2) installed.
*   Docker and Docker Compose v2 running.
*   *(Note for WSL2 users)*: If you encounter an `exec format error`, ensure you remove the `"credsStore": "desktop.exe"` line from your `~/.docker/config.json`.

### Build Steps
1.  Clone or create the project directory and ensure your `.env`, `docker-compose.yml`, and the two mock directories (`mock_prm` and `mock_trainer`) are present as configured.
2.  Open your WSL terminal and navigate to the project root.
3.  Execute the build command to compile the Python images and download the PyTorch/FastAPI dependencies:
    ```bash
    docker compose up -d --build
    ```
4.  Verify all four containers (`openclaw_agent`, `policy_server`, `prm_server`, `trainer_engine`) are running:
    ```bash
    docker compose ps
    ```
5.  Pull the required inference model into the empty Ollama container:
    ```bash
    docker exec -it policy_server ollama pull qwen2.5:1.5b
    ```

### Why Qwen2.5 0.5B is the best choice for your setup:

- CPU-Optimized: Your machine relies on an Intel CPU with no dedicated NVIDIA GPU. Models with billions of parameters will freeze your system. At only 0.5 billion parameters, Qwen is exceptionally lightweight and will run smoothly on your hardware.
- Low Memory Footprint: The quantized qwen2.5:0.5b model takes up less than 400 MB of RAM, leaving plenty of system resources for your Docker containers, WSL environment, and Windows host.
- Agentic Capabilities: Despite its tiny size, Qwen 2.5 possesses strong instruction-following capabilities and was specifically trained to understand structured outputs (like JSON) and code generation. This is critical, as OpenClaw relies on JSON to execute tool calls.

## 4. How to Use

### Accessing the Agent
1.  Open your host machine's web browser and navigate to `http://127.0.0.1:18789`.
2.  Authenticate using the `OPENCLAW_GATEWAY_TOKEN` defined in your `.env` file.
3.  In the bottom-right model selector, set the provider to **Ollama** and type `qwen2.5:1.5b` as the model name.
4.  Click the shield icon (Read-Only mode) in the prompt box and change it to allow workspace execution. 

### Monitoring the RL Loop
Because the architecture runs asynchronously, you must tail the logs of the individual containers to see the system working in real-time. Open a new terminal and run these commands to watch the pipeline:

*   **To watch the agent interact with the workspace:**
    ```bash
    docker logs -f openclaw_agent
    ```
*   **To watch Ollama process the context window and tokens:**
    ```bash
    docker logs -f policy_server
    ```
*   **To watch the PRM Judge evaluate actions and assign scores (+1.0 / -1.0):**
    ```bash
    docker logs -f prm_server
    ```
*   **To watch the Trainer Engine calculate PyTorch gradients on the CPU:**
    ```bash
    docker logs -f trainer_engine
    ```

## 5. Troubleshooting & Diagnostics

### Inference Route Error
If OpenClaw displays an inference route error banner:

* Check that the model is pulled inside the container using `docker exec -it policy_server ollama list`.
* Ensure `OLLAMA_HOST` is set to `http://policy_server:11434` (do not append `/v1`, as OpenAI-compatible routing breaks OpenClaw tool-calling).
* Ensure `OLLAMA_API_KEY=ollama-local` is defined in `.env` to satisfy OpenClaw auth checks.

### Role Confusion & Context Dilution

OpenClaw injects ~10,000 tokens of system prompts and tool schemas into every request:

* Models smaller than 1B parameters (`qwen2.5:0.5b`) will suffer from context saturation, hallucinate identities, or fail basic instructions.
* Upgrading to `qwen2.5:1.5b` or `qwen2.5:3b` resolves prompt retention issues while maintaining viable CPU execution speeds.

## First Results

TL;DR

The chat does not go very well; we can see the responses are bad

<img src="images/first chat with openclaw.png" style="width:650px;">

but we do prove that we have an online training

<img src="images/docker-policy-trainer.png" style="width:650px;">

(Assessement from Gemini)

Technically, your entire proof-of-concept stack is working: OpenClaw connects to Ollama over the native API (/api/chat returns 200 OK), and the training engine runs its asynchronous LoRA updates in the background.

The breakdown in the chat is caused by context overload on a 0.5B model.

What Is Happening:

- Context Dilution: As shown in the logs (task.n_tokens = 10659), OpenClaw injects a massive ~10,600-token system prompt detailing its internal architecture and cataloging 51 tools.

- Capacity Limit: A 0.5-billion parameter model does not have the attention capacity to retain basic conversational context (like who is "Nick" and who is "Rob") when buried under 10,000 tokens of agent instructions. It becomes confused and hallucinates role assignments.

- Read-Only Mode (_intentionally_): The shield icon at the bottom left indicates the agent is in Read Only mode, meaning tool execution and write operations are restricted in the session.

How to Fix This:

The fix is to swap the policy model to a slightly larger, instruction-tuned model that can track state across large agent prompts while remaining lightweight enough to run smoothly on CPU.

1. Pull a 1.5B or 3B Model
In your WSL terminal, pull qwen2.5:1.5b (or qwen2.5:3b if you have at least 8 GB of free system RAM):

```bash
docker exec -it policy_server ollama pull qwen2.5:1.5b
```

Verification: Wait for the layers to finish downloading and confirm Ollama outputs success.

2. Update the Model in the Web UI

Click the model dropdown at the bottom right of the chat (where it currently says qwen2.5:0.5b).

Switch it to qwen2.5:1.5b.

Start a fresh conversation thread using the + (New Chat) button so the context resets cleanly.

3. Enable Agent Actions (Optional)

If you want the agent to execute real commands and generate state signals for your PRM judge:

Click the Read Only shield button in the bottom-left prompt toolbar.

Switch the workspace execution permissions to allow the agent to run commands or write files inside its isolated Docker workspace.