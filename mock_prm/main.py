from fastapi import FastAPI, Request

app = FastAPI()


@app.post("/evaluate")
async def evaluate(request: Request):
    data = await request.json()
    action = data.get("action", "unknown_action")

    print("\n--- [PRM SERVER] NEW REQUEST RECEIVED ---")
    print(f"[PRM SERVER] Analyzing agent action: '{action}'")

    # Simple logic: positive reward unless an error occurred
    if "error" not in str(data).lower():
        reward = 1.0
        print("[PRM SERVER] Result: SUCCESS. Assigning Reward: +1.0")
    else:
        reward = -1.0
        print("[PRM SERVER] Result: FAILURE (Error detected). Assigning Reward: -1.0")

    print("--- [PRM SERVER] EVALUATION COMPLETE ---\n")
    return {"reward": reward, "status": "success"}
