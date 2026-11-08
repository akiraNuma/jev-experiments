"""CLIへ渡したモデルと、応答に記録されたモデルを別に扱う。"""


def model_id(model):
    return {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}.get(model, model)


def model_metadata(response, requested_model):
    usage = response.get("modelUsage")
    return {"requested_model": requested_model,
            "models_used": list(usage) if isinstance(usage, dict) else [],
            "model_usage": usage}
