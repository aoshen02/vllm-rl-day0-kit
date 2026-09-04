#!/usr/bin/env python3
import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import aiohttp

INVALID = -9999999
INT64_MAX = (1 << 63) - 1


def get_answer_value(answer: str) -> int:
    numbers = re.findall(r"\d+", answer.replace(",", ""))
    if not numbers:
        return INVALID
    value = int(numbers[-1])
    return value if value <= INT64_MAX else INVALID


def build_gsm8k_prompts(
    dataset_dir: Path, num_questions: int, num_shots: int
) -> tuple[list[str], list[int]]:
    def load(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    train = load(dataset_dir / "train.jsonl")
    if len(train) < num_shots:
        raise ValueError("GSM8K train split has fewer rows than --num-shots")
    test = load(dataset_dir / "test.jsonl")[:num_questions]
    examples = "".join(
        f"Question: {item['question']}\nAnswer: {item['answer']}\n\n"
        for item in train[:num_shots]
    )
    prompts = [examples + f"Question: {item['question']}\nAnswer:" for item in test]
    labels = [get_answer_value(item["answer"]) for item in test]
    if len(prompts) != num_questions or INVALID in labels:
        raise ValueError("GSM8K dataset does not satisfy the requested contract")
    return prompts, labels


async def evaluate(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    prompts, labels = build_gsm8k_prompts(
        args.dataset_dir, args.num_questions, args.num_shots
    )
    records: list[dict] = [{} for _ in prompts]
    timeout = aiohttp.ClientTimeout(total=1800)
    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def request_one(index: int) -> None:
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompts[index]}],
                "temperature": 0.0,
                "max_tokens": args.max_tokens,
                "seed": 42,
            }
            try:
                async with (
                    semaphore,
                    session.post(
                        f"{args.host}:{args.port}/v1/chat/completions", json=payload
                    ) as response,
                ):
                    response.raise_for_status()
                    body = await response.json()
                choice = body["choices"][0]
                message = choice["message"]
                content = message.get("content") or ""
                reasoning_content = message.get("reasoning_content") or ""
                if not isinstance(content, str) or not isinstance(
                    reasoning_content, str
                ):
                    raise TypeError("chat response text must be a string")
                score_text = "\n".join(filter(None, (reasoning_content, content)))
                records[index] = {
                    "index": index,
                    "response_id": body.get("id"),
                    "label": labels[index],
                    "prediction": get_answer_value(score_text),
                    "content": content,
                    "reasoning_content": reasoning_content,
                    "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": body.get("usage", {}).get(
                        "completion_tokens", 0
                    ),
                }
            except Exception as error:
                records[index] = {
                    "index": index,
                    "label": labels[index],
                    "prediction": INVALID,
                    "error": f"{type(error).__name__}: {error}",
                    "completion_tokens": 0,
                }

        started = time.perf_counter()
        await asyncio.gather(*(request_one(i) for i in range(len(prompts))))
        latency = time.perf_counter() - started

    correct_count = sum(
        record["prediction"] == label for record, label in zip(records, labels)
    )
    total_tokens = sum(record["completion_tokens"] for record in records)
    request_error_count = sum("error" in record for record in records)
    summary = {
        "accuracy": correct_count / len(records),
        "correct_count": correct_count,
        "invalid_rate": sum(record["prediction"] == INVALID for record in records)
        / len(records),
        "latency": latency,
        "questions_per_second": len(records) / latency,
        "total_output_tokens": total_tokens,
        "request_error_count": request_error_count,
        "tokens_per_second": total_tokens / latency,
        "num_questions": len(records),
        "num_shots": args.num_shots,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": 42,
        "concurrency": args.concurrency,
        "connection_limit": args.concurrency,
        "scored_text": "reasoning_content followed by content",
    }
    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--num-questions", type=int, default=1319)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=1319)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--details", required=True, type=Path)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.num_questions < 1:
        parser.error("--num-questions must be at least 1")
    if args.num_shots < 0:
        parser.error("--num-shots must be non-negative")
    summary, records = asyncio.run(evaluate(args))
    with args.output.open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, sort_keys=True)
    with args.details.open("w", encoding="utf-8") as details:
        for record in records:
            details.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["request_error_count"]:
        raise SystemExit(f"GSM8K request failures: {summary['request_error_count']}")


if __name__ == "__main__":
    main()
