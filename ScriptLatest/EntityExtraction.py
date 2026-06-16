import json
import os
import time
import csv
from dataclasses import dataclass
from typing import List, Set, Dict, Any, Tuple

from getpass import getpass

from openai import OpenAI


# Paths for persisted experiment artifacts
RESULTS_JSON_PATH = "ab_results.json"
DASHBOARD_HTML_PATH = "ab_dashboard.html"
RESULTS_CSV_PATH = "ab_results.csv"

# Optional pricing config: cost in USD per 1K tokens.
# Fill in values for the models you use to get cost estimates.
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # Example (update with actual pricing for your account as needed):
    # "gpt-4.1-mini": {"input": 0.00015, "output": 0.00060},
    # "gpt-4.1": {"input": 0.00500, "output": 0.01500},
}


@dataclass
class ExtractionResult:
    model: str
    entities: Set[str]
    numbers: Set[str]
    latency_seconds: float
    raw_response: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float


def ensure_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = getpass("Enter your OpenAI API key: ").strip()
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        os.environ["OPENAI_API_KEY"] = api_key
    return api_key


def create_client() -> OpenAI:
    ensure_api_key()
    return OpenAI()


def normalize_entity(value: str) -> str:
    """Normalize entity strings for comparison."""
    value = value.strip().lower()
    # Strip simple surrounding punctuation
    value = value.strip(".,;:!?\"'()[]{}")
    # Collapse internal whitespace
    value = " ".join(value.split())
    return value


def parse_taxonomy_file(path: str) -> Set[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Taxonomy file not found: {path}")

    entities: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on commas / semicolons if present
            parts = []
            for chunk in line.split(";"):
                parts.extend(chunk.split(","))
            for part in parts:
                norm = normalize_entity(part)
                if norm:
                    entities.add(norm)
    return entities


def read_document(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Document file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def list_openai_models(client: OpenAI) -> List[str]:
    models = client.models.list()
    model_ids = sorted(m.id for m in models.data)
    # Filter to common chat-capable models first, but keep full list
    preferred_prefixes = ("gpt", "o1", "o3")
    preferred = [m for m in model_ids if m.startswith(preferred_prefixes)]
    others = [m for m in model_ids if m not in preferred]
    return preferred + others


def openai_api_setup() -> Tuple[OpenAI, List[str]]:
    """
    Step 1: OpenAI API Setup

    - Uses API key from env / prompt.
    - Validates by listing available models.
    """
    print("Step 1: OpenAI API Setup")
    client = create_client()
    model_ids = list_openai_models(client)
    if not model_ids:
        raise RuntimeError("No OpenAI models available for this API key.")
    print(f"Authenticated successfully. {len(model_ids)} models available.")
    return client, model_ids


def prompt_for_models(model_ids: List[str]) -> Tuple[str, str]:
    print("\nAvailable OpenAI models (showing all from your account):")
    for mid in model_ids:
        print(f"- {mid}")

    def pick(prompt_text: str) -> str:
        while True:
            raw = input(prompt_text).strip()
            if not raw:
                print("Please enter a non-empty model name (for example: gpt-4.1-mini).")
                continue
            if raw in model_ids:
                return raw
            print("That model name is not in the list above or you do not have access to it.")
            print("Please copy-paste one of the model names exactly as shown.")

    model_a = pick("Enter OpenAI model name for Model A (e.g., gpt-4.1-mini): ")
    model_b = pick("Enter OpenAI model name for Model B (e.g., gpt-4.1): ")
    return model_a, model_b


def prompt_for_float(prompt_text: str, default: float) -> float:
    while True:
        raw = input(f"{prompt_text} (default {default}): ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")


def estimate_token_count(text: str) -> int:
    """Rough token estimate assuming ~4 characters per token."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def validate_token_limits(text: str, max_context_tokens: int = 128_000) -> None:
    estimated = estimate_token_count(text)
    print(f"Estimated tokens for input text: ~{estimated} (context limit ≈ {max_context_tokens}).")
    if estimated > max_context_tokens:
        print("Warning: Input may exceed typical context limits for some models.")


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate cost in USD based on MODEL_PRICING (per 1K tokens).
    If the model is not configured, returns 0.0.
    """
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0
    input_rate = pricing.get("input", 0.0)
    output_rate = pricing.get("output", 0.0)
    return (input_tokens / 1000.0) * input_rate + (output_tokens / 1000.0) * output_rate


def model_supports_sampling_params(model: str) -> bool:
    """
    Some models (notably reasoning models like o1/o3) do not support custom
    temperature/top_p and will only accept their defaults. For those models we
    must NOT send sampling params or the API returns an error.
    """
    lowered = model.lower()
    # Adjust this list if OpenAI introduces more models with fixed sampling.
    return not (lowered.startswith("o1") or lowered.startswith("o3"))


def generate_task_prompt(
    task_type: str,
    taxonomy_entities: Set[str],
    document: str,
    length_hint: str = "",
    topic: str = "",
    output_format: str = "text",
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Step 3: Prompt Construction (generate_task_prompt)

    Supports:
    - entity_extraction (primary for this experiment)
    - summarization (skeleton)
    - generation (skeleton)
    """
    if task_type == "entity_extraction":
        taxonomy_list = sorted(taxonomy_entities)
        system_prompt = (
            "You are an information extraction engine.\n"
            "Your goal is STRICT TAXONOMY-AWARE entity extraction.\n\n"
            "You are given a reference taxonomy (a list of canonical entities) and a document.\n"
            "- You MUST only select entities from this taxonomy list that are actually mentioned\n"
            "  in the document (exactly or with trivial case/whitespace differences).\n"
            "- You MUST NOT invent new entities or output strings that are not EXACT entries\n"
            "  from the taxonomy list.\n"
            "- If the document mentions something that is not clearly one of the taxonomy\n"
            "  entities, you MUST ignore it.\n"
            "- If no taxonomy entities are present, return an empty list for \"entities\".\n\n"
            "Return ONLY a strict JSON object with this exact structure:\n"
            "{\n"
            '  \"entities\": [\"taxonomy_entity_1\", \"taxonomy_entity_2\", ...],\n'
            '  \"numbers\": [\"42\", \"3.14\", ...]\n'
            "}\n"
            "- Every string in \"entities\" MUST be exactly equal to one of the taxonomy\n"
            "  entries provided (after trimming spaces).\n"
            "- Do not include explanations or any extra text."
        )
        user_prompt = (
            "Reference taxonomy (entities). Each entry is a canonical entity string.\n"
            "You may ONLY return entities from this list:\n"
            f"{json.dumps(taxonomy_list, ensure_ascii=False)}\n\n"
            "Document to analyze:\n"
            f"{document}"
        )
        response_format = {"type": "json_object"}
        return system_prompt, user_prompt, response_format

    if task_type == "summarization":
        system_prompt = "You are a helpful assistant that summarizes text clearly and concisely."
        length_text = length_hint or "a concise"
        user_prompt = (
            f"Summarize the following text in {length_text} length.\n"
            f"Desired formatting: {output_format}.\n\n"
            f"Text:\n{document}"
        )
        return system_prompt, user_prompt, {}

    if task_type == "generation":
        system_prompt = "You are a helpful assistant that writes clear, well-structured content."
        length_text = length_hint or "a concise"
        topic_text = topic or "the given topic"
        user_prompt = (
            f"Write about {topic_text} in {length_text} length.\n"
            f"Desired formatting: {output_format}.\n\n"
            f"Source text (optional context):\n{document}"
        )
        return system_prompt, user_prompt, {}

    # Fallback to entity_extraction behavior if unknown task
    return generate_task_prompt(
        task_type="entity_extraction",
        taxonomy_entities=taxonomy_entities,
        document=document,
        length_hint=length_hint,
        topic=topic,
        output_format=output_format,
    )


def derive_taxonomy_from_source(
    client: OpenAI,
    model: str,
    source_document: str,
) -> Set[str]:
    """
    Derive a taxonomy set from a source document using an OpenAI model.

    This is useful when File 1 is a rich "source" document instead of an
    explicit taxonomy list. The model is asked to extract a deduplicated list
    of canonical entity names.
    """
    system_prompt = (
        "You are building a canonical taxonomy of entities from a source document.\n"
        "Extract a deduplicated list of important entities (names, concepts, key\n"
        "fields) that are explicitly mentioned in the document.\n\n"
        "Rules:\n"
        "- Use short, canonical names (e.g., \"employment history\", \"full name\").\n"
        "- Do NOT invent entities that are not clearly present.\n"
        "- Avoid full sentences; each item should be a concise label.\n"
        "- Deduplicate semantically identical items; keep one canonical form.\n\n"
        "Return ONLY a strict JSON object of the form:\n"
        "{ \"taxonomy_entities\": [\"entity1\", \"entity2\", ...] }\n"
        "No explanations or extra text."
    )
    user_prompt = (
        "Source document:\n"
        f"{source_document}"
    )

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        top_p=1.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    end = time.perf_counter()
    _ = end - start  # latency not currently used here

    content = response.choices[0].message.content or "{}"
    try:
        parsed: Dict[str, Any] = json.loads(content)
    except json.JSONDecodeError:
        first_brace = content.find("{")
        last_brace = content.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            snippet = content[first_brace : last_brace + 1]
            parsed = json.loads(snippet)
        else:
            raise ValueError(f"Taxonomy derivation did not return valid JSON: {content!r}")

    raw_entities = parsed.get("taxonomy_entities", []) or []
    taxonomy: Set[str] = set()
    for e in raw_entities:
        if isinstance(e, str):
            norm = normalize_entity(e)
            if norm:
                taxonomy.add(norm)
    return taxonomy


def extract_entities_with_model(
    client: OpenAI,
    model: str,
    taxonomy_entities: Set[str],
    document: str,
    temperature: float,
    top_p: float,
) -> ExtractionResult:
    system_prompt, user_prompt, response_format = generate_task_prompt(
        task_type="entity_extraction",
        taxonomy_entities=taxonomy_entities,
        document=document,
    )

    start = time.perf_counter()
    request_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": response_format,
    }
    # Only send temperature/top_p for models that support them.
    if model_supports_sampling_params(model):
        # Some models only support the default temperature; treat 0.0 as
        # "use model default" and omit the parameter entirely.
        if temperature is not None and temperature != 0.0:
            request_kwargs["temperature"] = temperature
        request_kwargs["top_p"] = top_p

    response = client.chat.completions.create(**request_kwargs)
    end = time.perf_counter()

    latency = end - start

    # Token usage and cost
    usage = getattr(response, "usage", None)
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    if usage is not None:
        # Support both old and new field names defensively
        input_tokens = getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or (input_tokens + output_tokens)
    cost_usd = estimate_cost_usd(model, input_tokens, output_tokens)

    content = response.choices[0].message.content or "{}"
    try:
        parsed: Dict[str, Any] = json.loads(content)
    except json.JSONDecodeError:
        # Best-effort fallback: try to locate JSON object within content
        first_brace = content.find("{")
        last_brace = content.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            snippet = content[first_brace : last_brace + 1]
            parsed = json.loads(snippet)
        else:
            raise ValueError(f"Model {model} did not return valid JSON: {content!r}")

    raw_entities = parsed.get("entities", []) or []
    raw_numbers = parsed.get("numbers", []) or []

    entities_norm: Set[str] = set()
    for e in raw_entities:
        if isinstance(e, str):
            norm = normalize_entity(e)
            if norm:
                entities_norm.add(norm)

    numbers_norm: Set[str] = set()
    for n in raw_numbers:
        if isinstance(n, str):
            norm = normalize_entity(n)
            if norm:
                numbers_norm.add(norm)

    return ExtractionResult(
        model=model,
        entities=entities_norm,
        numbers=numbers_norm,
        latency_seconds=latency,
        raw_response=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
    )


def compute_metrics(
    gold_entities: Set[str],
    predicted_entities: Set[str],
) -> Dict[str, float]:
    """
    Compute core taxonomy-aware metrics:
    - recall  (entity_extraction_ratio): how many gold entities we recovered
    - precision (relevancy): how many predicted entities are actually gold
    - f1: harmonic mean of precision and recall

    For backwards compatibility, we also expose:
    - accuracy  == recall
    - relevancy == precision
    - entities_matching_score == f1
    """
    if not gold_entities:
        return {
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "entity_extraction_ratio": 0.0,
            "accuracy": 0.0,
            "relevancy": 0.0,
            "entities_matching_score": 0.0,
        }

    tp = len(gold_entities & predicted_entities)
    fp = len(predicted_entities - gold_entities)
    fn = len(gold_entities - predicted_entities)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        # New, clearer names
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "entity_extraction_ratio": recall,
        # Backwards-compatible aliases
        "accuracy": recall,
        "relevancy": precision,
        "entities_matching_score": f1,
    }


def aggregate_results(
    taxonomy_entities: Set[str],
    result_a: ExtractionResult,
    result_b: ExtractionResult,
    metrics_a: Dict[str, float],
    metrics_b: Dict[str, float],
) -> Dict[str, Any]:
    """
    Step 6: Result Aggregation

    - Combine metrics, latency, token counts, and cost.
    - Identify winning model based on entities_matching_score (F1), then accuracy.
    """
    gold_all = set(taxonomy_entities)

    def model_payload(label: str, res: ExtractionResult, metrics: Dict[str, float]) -> Dict[str, Any]:
        matched = sorted(res.entities & gold_all)
        missing = sorted(gold_all - res.entities)
        extras = sorted(res.entities - gold_all)
        return {
            "label": label,
            "model_name": res.model,
            "latency_seconds": res.latency_seconds,
            "metrics": metrics,
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "total_tokens": res.total_tokens,
            "cost_usd": res.cost_usd,
            "extracted_entities_count": len(res.entities),
            "extracted_numbers_count": len(res.numbers),
            "matched_entities_count": len(matched),
            "matched_entities_sample": matched[:10],
            "missing_entities_count": len(missing),
            "missing_entities_sample": missing[:20],
            "extra_entities_count": len(extras),
            "extra_entities_sample": extras[:20],
            # Full lists for JSON consumers; can be large for big taxonomies.
            "missing_entities": missing,
            "extra_entities": extras,
        }

    model_a_payload = model_payload("Model A", result_a, metrics_a)
    model_b_payload = model_payload("Model B", result_b, metrics_b)

    # Determine winner by F1, then recall, then lower latency
    def effectiveness_key(p: Dict[str, Any]) -> Tuple[float, float, float]:
        m = p["metrics"]
        return (
            m.get("f1", m.get("entities_matching_score", 0.0)),
            m.get("recall", m.get("accuracy", 0.0)),
            -float(p.get("latency_seconds", 0.0)),
        )

    models = [model_a_payload, model_b_payload]
    winner = max(models, key=effectiveness_key)

    return {
        "task_type": "entity_extraction",
        "taxonomy_entity_count": len(gold_all),
        "models": models,
        "winner": winner,
    }


def save_results_artifacts(aggregated: Dict[str, Any]) -> None:
    """
    Steps 7 & 8: Reporting & Visualization / Result Visualization

    - Save JSON summary.
    - Save CSV summary.
    - Generate an interactive HTML dashboard with charts.
    """
    # JSON
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)

    # CSV (simple per-model summary)
    headers = [
        "label",
        "model_name",
        "latency_seconds",
        "recall",                     # entity_extraction_ratio
        "precision",                  # relevancy
        "f1",
        "entity_extraction_ratio",
        "taxonomy_entities_count",    # entities in Document 1 (ground truth)
        "extracted_entities_count",   # entities in Document 2 (per model)
        "matched_entities_count",     # overlap between Doc1 and Doc2
        "missing_entities_count",
        "extra_entities_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_usd",
        "extracted_numbers_count",
    ]

    with open(RESULTS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for m in aggregated["models"]:
            metrics = m["metrics"]
            writer.writerow(
                [
                    m["label"],
                    m["model_name"],
                    f"{m['latency_seconds']:.3f}",
                    f"{metrics.get('recall', 0.0):.4f}",
                    f"{metrics.get('precision', 0.0):.4f}",
                    f"{metrics.get('f1', 0.0):.4f}",
                    f"{metrics.get('entity_extraction_ratio', 0.0):.4f}",
                    aggregated.get("taxonomy_entity_count", 0),
                    m["extracted_entities_count"],
                    m["matched_entities_count"],
                    m.get("missing_entities_count", 0),
                    m.get("extra_entities_count", 0),
                    m["input_tokens"],
                    m["output_tokens"],
                    m["total_tokens"],
                    f"{m['cost_usd']:.6f}",
                    m["extracted_numbers_count"],
                ]
            )

    # HTML dashboard with charts via Chart.js
    models = aggregated["models"]
    winner = aggregated.get("winner")
    model_labels = [m["label"] for m in models]
    accuracies = [m["metrics"].get("recall", 0.0) for m in models]      # recall/entity_extraction_ratio
    relevancies = [m["metrics"].get("precision", 0.0) for m in models]  # precision
    f1_scores = [m["metrics"].get("f1", 0.0) for m in models]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>A/B Entity Extraction Experiment Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {{
      --bg: #f5f7fb;
      --card-bg: #ffffff;
      --accent: #1f7ae0;
      --accent-soft: #e3f0ff;
      --border: #dde3f0;
      --text-main: #1f2933;
      --text-muted: #6b778d;
      --success: #2e7d32;
      --danger: #c62828;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text-main);
    }}
    h1, h2, h3 {{
      margin: 0;
      font-weight: 600;
    }}
    a {{
      color: var(--accent);
    }}
    .page {{
      max-width: 1200px;
      margin: 0 auto 40px auto;
    }}
    .header {{
      margin-bottom: 24px;
      padding: 16px 20px;
      border-radius: 12px;
      background: linear-gradient(135deg, #1f7ae0, #7759ff);
      color: #fff;
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
    }}
    .header-main {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 16px;
    }}
    .header-sub {{
      margin-top: 8px;
      font-size: 13px;
      display: flex;
      gap: 24px;
      flex-wrap: wrap;
      color: rgba(255, 255, 255, 0.9);
    }}
    .badge-winner {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.16);
      font-size: 12px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      margin: 20px 0 28px 0;
    }}
    .summary-card {{
      background: var(--card-bg);
      border-radius: 12px;
      padding: 14px 16px;
      border: 1px solid var(--border);
      box-shadow: 0 4px 10px rgba(15, 23, 42, 0.06);
    }}
    .summary-card.winner {{
      border-color: var(--accent);
      box-shadow: 0 6px 16px rgba(31, 122, 224, 0.18);
    }}
    .summary-card-title {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
      font-size: 13px;
      color: var(--text-muted);
    }}
    .summary-card-model {{
      font-size: 13px;
      color: var(--text-muted);
    }}
    .metric-row {{
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      margin-top: 2px;
    }}
    .metric-label {{
      color: var(--text-muted);
    }}
    .metric-value-strong {{
      font-weight: 600;
    }}
    .metric-value-good {{
      color: var(--success);
      font-weight: 600;
    }}
    .metric-value-bad {{
      color: var(--danger);
      font-weight: 600;
    }}
    .charts {{
      display: flex;
      flex-wrap: wrap;
      gap: 24px;
      margin-bottom: 24px;
    }}
    .chart-container {{
      width: 320px;
      height: 320px;
      background: var(--card-bg);
      border-radius: 12px;
      padding: 12px 14px 16px 14px;
      border: 1px solid var(--border);
      box-shadow: 0 4px 10px rgba(15, 23, 42, 0.06);
    }}
    .chart-container h2 {{
      font-size: 15px;
      margin-bottom: 6px;
    }}
    .section-title {{
      font-size: 16px;
      margin: 18px 0 8px 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 13px;
      background: var(--card-bg);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 4px 10px rgba(15, 23, 42, 0.06);
    }}
    th, td {{
      padding: 8px 10px;
      text-align: right;
      border-bottom: 1px solid #edf0f7;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    thead th {{
      background: var(--accent-soft);
      color: var(--text-main);
      font-weight: 600;
    }}
    tbody tr:nth-child(even) {{
      background: #f8fafc;
    }}
    .footer {{
      margin-top: 16px;
      font-size: 12px;
      color: var(--text-muted);
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div class="header-main">
        <h1>A/B Entity Extraction Experiment Dashboard</h1>
        {"<span class='badge-winner'>Winner: " + winner["label"] + " (" + winner["model_name"] + ")</span>" if winner else ""}
      </div>
      <div class="header-sub">
        <span>Task type: {aggregated.get("task_type", "")}</span>
        <span>Taxonomy entity count: {aggregated.get("taxonomy_entity_count", 0)}</span>
      </div>
    </div>

    <div class="summary-grid">
"""
    for m in models:
        metrics = m["metrics"]
        is_winner = winner and m["label"] == winner["label"]
        html += f"""
      <div class="summary-card{' winner' if is_winner else ''}">
        <div class="summary-card-title">
          <div><strong>{m["label"]}</strong></div>
          <div class="summary-card-model">{m["model_name"]}</div>
        </div>
        <div class="metric-row">
          <span class="metric-label">F1 (entity score)</span>
          <span class="metric-value-strong">{metrics.get("f1", 0.0):.3f}</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Recall</span>
          <span class="metric-value-strong">{metrics.get("recall", 0.0):.3f}</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Precision</span>
          <span class="metric-value-strong">{metrics.get("precision", 0.0):.3f}</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Matched / Missing / Extra</span>
          <span class="metric-value-strong">{m["matched_entities_count"]} / {m.get("missing_entities_count", 0)} / {m.get("extra_entities_count", 0)}</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Latency (s)</span>
          <span>{m["latency_seconds"]:.3f}</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Total tokens</span>
          <span>{m["total_tokens"]}</span>
        </div>
      </div>
"""
    html += """
    </div>

    <div class="charts">
    <div class="chart-container">
      <h2>Recall (Entity extraction ratio)</h2>
      <canvas id="accuracyChart"></canvas>
    </div>
    <div class="chart-container">
      <h2>Precision (Relevancy)</h2>
      <canvas id="relevancyChart"></canvas>
    </div>
    <div class="chart-container">
      <h2>Entity score (F1)</h2>
      <canvas id="f1Chart"></canvas>
    </div>
    </div>

    <h2 class="section-title">Model Outputs, Entity Statistics & Cost Summary</h2>
    <table>
    <thead>
      <tr>
        <th>Label</th>
        <th>Model</th>
        <th>Latency (s)</th>
        <th>Recall (entity extraction ratio)</th>
        <th>Precision (relevancy)</th>
        <th>F1 (entity score)</th>
        <th>Entities in Doc 1 (taxonomy)</th>
        <th>Entities in Doc 2 (extracted)</th>
        <th>Overlap (Doc1 ∩ Doc2)</th>
        <th>Missing entities (Doc1 − Doc2)</th>
        <th>Extra entities (Doc2 − Doc1)</th>
        <th>Total Tokens</th>
        <th>Cost (USD)</th>
      </tr>
    </thead>
    <tbody>
"""
    for m in models:
        metrics = m["metrics"]
        html += f"""      <tr>
        <td>{m["label"]}</td>
        <td>{m["model_name"]}</td>
        <td>{m["latency_seconds"]:.3f}</td>
        <td>{metrics.get("recall", 0.0):.3f}</td>
        <td>{metrics.get("precision", 0.0):.3f}</td>
        <td>{metrics.get("f1", 0.0):.3f}</td>
        <td>{aggregated.get("taxonomy_entity_count", 0)}</td>
        <td>{m["extracted_entities_count"]}</td>
        <td>{m["matched_entities_count"]}</td>
        <td>{m.get("missing_entities_count", 0)}</td>
        <td>{m.get("extra_entities_count", 0)}</td>
        <td>{m["total_tokens"]}</td>
        <td>{m["cost_usd"]:.6f}</td>
      </tr>
"""

    html += f"""    </tbody>
  </table>

  <h2 class="section-title">Common Entities between Doc 1 and Doc 2</h2>
  <table>
    <thead>
      <tr>
        <th>Label</th>
        <th>Model</th>
        <th>Common entities (Doc1 ∩ Doc2, up to 20)</th>
      </tr>
    </thead>
    <tbody>
"""
    for m in models:
        common_entities = ", ".join(m.get("matched_entities_sample", [])[:20])
        html += f"""      <tr>
        <td>{m["label"]}</td>
        <td>{m["model_name"]}</td>
        <td style="text-align:left">{common_entities}</td>
      </tr>
"""

    html += f"""    </tbody>
  </table>

  <p class="footer">
    <a href="{RESULTS_CSV_PATH}" download>Download CSV export</a> ·
    All scores are computed against the same taxonomy for both models.
  </p>

  </div> <!-- /page -->

  <script>
    const labels = {json.dumps(model_labels)};
    const accuracies = {json.dumps(accuracies)};
    const relevancies = {json.dumps(relevancies)};
    const f1Scores = {json.dumps(f1_scores)};

    function createPieChart(ctxId, label, data) {{
      const ctx = document.getElementById(ctxId).getContext('2d');
      new Chart(ctx, {{
        type: 'pie',
        data: {{
          labels: labels,
          datasets: [{{
            label: label,
            data: data,
            backgroundColor: ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'],
          }}],
        }},
      }});
    }}

    createPieChart('accuracyChart', 'Recall', accuracies);
    createPieChart('relevancyChart', 'Precision', relevancies);
    createPieChart('f1Chart', 'F1', f1Scores);
  </script>
</body>
</html>
"""

    with open(DASHBOARD_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)


def run_experiment() -> None:
    print("=== A/B Experiment: Entity Extraction with OpenAI ===")

    # Step 1: OpenAI API Setup
    client, model_ids = openai_api_setup()

    # Step 2: Get Inputs from user (get_user_inputs)
    print("\nStep 2: Get Inputs from user (entity extraction task)")
    taxonomy_path = input("Enter path to source/taxonomy file (File 1): ").strip()
    document_path = input("Enter path to summary/reference document (File 2): ").strip()

    # Choose models up front so we can optionally use Model A to derive taxonomy
    model_a, model_b = prompt_for_models(model_ids)
    print(f"\nSelected models:\n- Model A: {model_a}\n- Model B: {model_b}")

    # Decide whether File 1 is a ready-made taxonomy list or a source document
    mode = input(
        "Is File 1 a taxonomy list or a source document to derive the taxonomy from?\n"
        "Enter 'list' for a ready taxonomy list, or 'source' to derive taxonomy from File 1 "
        "using Model A. [list/source] (default: list): "
    ).strip().lower() or "list"

    if mode not in {"list", "source"}:
        print("Unrecognized choice, defaulting to 'list' mode.")
        mode = "list"

    if mode == "list":
        taxonomy_entities = parse_taxonomy_file(taxonomy_path)
    else:
        print("\nDeriving taxonomy entities from source document (File 1) using Model A...")
        source_text = read_document(taxonomy_path)
        validate_token_limits(source_text)
        taxonomy_entities = derive_taxonomy_from_source(client, model_a, source_text)

    document_text = read_document(document_path)

    validate_token_limits(document_text)
    print(f"Loaded {len(taxonomy_entities)} unique taxonomy entities.")

    # Inform user if any selected model does not support custom sampling params.
    for label, m in (("Model A", model_a), ("Model B", model_b)):
        if not model_supports_sampling_params(m):
            print(
                f"Note: {label} model '{m}' does not support custom temperature/top_p; "
                "the API defaults will be used for it."
            )

    temperature = prompt_for_float("Temperature (enter 0 to use model default)", 0.0)
    top_p = prompt_for_float("top_p", 1.0)

    # Step 3 is implemented via generate_task_prompt, which is used inside extract_entities_with_model.

    # Step 4: Model Execution
    print("\nStep 4: Model Execution")
    print("\nRunning extraction with Model A...")
    result_a = extract_entities_with_model(
        client=client,
        model=model_a,
        taxonomy_entities=taxonomy_entities,
        document=document_text,
        temperature=temperature,
        top_p=top_p,
    )

    print("Running extraction with Model B...")
    result_b = extract_entities_with_model(
        client=client,
        model=model_b,
        taxonomy_entities=taxonomy_entities,
        document=document_text,
        temperature=temperature,
        top_p=top_p,
    )

    # Step 5: Output Evaluation
    # Combine entities and numbers for evaluation
    gold_all: Set[str] = set(taxonomy_entities)
    # Numbers from taxonomy are not explicitly provided; evaluation is primarily on entities.

    pred_a_all = set(result_a.entities)
    pred_b_all = set(result_b.entities)
    metrics_a = compute_metrics(gold_all, pred_a_all)
    metrics_b = compute_metrics(gold_all, pred_b_all)

    print("\n=== Step 5: Output Evaluation Results ===")
    print("\nGround-truth entity count:", len(gold_all))

    def print_model_result(label: str, res: ExtractionResult, metrics: Dict[str, float]) -> None:
        print(f"\n--- {label} ({res.model}) ---")
        print(f"Latency: {res.latency_seconds:.3f} seconds")
        print(f"Extracted entities: {len(res.entities)}")
        print(f"Extracted numbers: {len(res.numbers)}")
        print(f"Recall (entity extraction ratio): {metrics['recall']:.3f}")
        print(f"Precision (relevancy): {metrics['precision']:.3f}")
        print(f"Entity score (F1): {metrics['f1']:.3f}")
        print(f"Total tokens: {res.total_tokens} (input: {res.input_tokens}, output: {res.output_tokens})")
        print(f"Estimated cost (USD): {res.cost_usd:.6f}")
        # Show all extracted taxonomy entities so the user can inspect for hallucinations.
        if res.entities:
            print("Extracted taxonomy entities (Doc 2):")
            for e in sorted(res.entities):
                print(f"  - {e}")
        else:
            print("Extracted taxonomy entities (Doc 2): NONE")
        matched = sorted(res.entities & gold_all)
        missing = sorted(gold_all - res.entities)
        extras = sorted(res.entities - gold_all)
        print("Matched entities vs taxonomy (up to 10):")
        for e in matched[:10]:
            print(f"  - {e}")
        print("Missing entities (Doc1 − Doc2, up to 10):")
        for e in missing[:10]:
            print(f"  - {e}")
        if not missing:
            print("  (none)")
        print("Extra entities (Doc2 − Doc1, up to 10):")
        for e in extras[:10]:
            print(f"  - {e}")
        if not extras:
            print("  (none)")

    print_model_result("Model A", result_a, metrics_a)
    print_model_result("Model B", result_b, metrics_b)

    # Step 6: Result Aggregation
    aggregated = aggregate_results(
        taxonomy_entities,
        result_a,
        result_b,
        metrics_a,
        metrics_b,
    )

    # Steps 7 & 8: Reporting, Visualization & Result Visualization
    save_results_artifacts(aggregated)

    # Step 9: Completion
    winner = aggregated["winner"]
    print("\n=== Step 9: Completion ===")
    print(
        f"Winning model: {winner['label']} ({winner['model_name']}) "
        f"with F1={winner['metrics']['f1']:.3f}, "
        f"Recall={winner['metrics']['recall']:.3f}, "
        f"Precision={winner['metrics']['precision']:.3f}"
    )
    print(f"\nResults JSON saved to: {RESULTS_JSON_PATH}")
    print(f"CSV export saved to: {RESULTS_CSV_PATH}")
    print(f"Dashboard HTML saved to: {DASHBOARD_HTML_PATH}")


if __name__ == "__main__":
    run_experiment()


