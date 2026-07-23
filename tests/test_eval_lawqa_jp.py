import json

from scripts.eval_lawqa_jp import (
    LawqaEvalResult,
    build_lawqa_question,
    extract_answer_label,
    load_lawqa_samples,
    summarize_results,
)


def test_load_lawqa_samples_reads_japanese_fields(tmp_path) -> None:
    path = tmp_path / "lawqa.json"
    path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "ファイル名": "sample-1",
                        "コンテキスト": "根拠本文",
                        "問題文": "正しいものはどれですか。",
                        "指示": "選択肢から選んでください。",
                        "選択肢": "a 誤り\nb 正しい\nc 誤り\nd 誤り",
                        "output": "b",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    samples = load_lawqa_samples(path)

    assert len(samples) == 1
    assert samples[0].filename == "sample-1"
    assert samples[0].answer_label == "b"


def test_build_lawqa_question_includes_choices_and_strict_answer_instruction() -> None:
    sample = load_lawqa_samples_from_dict()

    question = build_lawqa_question(sample)

    assert "問題文:" in question
    assert "選択肢:" in question
    assert "a, b, c, d" in question


def test_extract_answer_label_handles_common_formats() -> None:
    assert extract_answer_label("c") == "c"
    assert extract_answer_label("回答: b です。") == "b"
    assert extract_answer_label("正解は d です。") == "d"
    assert extract_answer_label("選択肢aが正しいです。") == "a"
    assert extract_answer_label("判断できません。") is None


def test_summarize_results_calculates_accuracy() -> None:
    results = [
        LawqaEvalResult(
            filename="one",
            expected_label="a",
            predicted_label="a",
            retrieval_hit=True,
            source_count=3,
            answer="a",
        ),
        LawqaEvalResult(
            filename="two",
            expected_label="b",
            predicted_label="c",
            retrieval_hit=False,
            source_count=3,
            answer="c",
        ),
    ]

    summary = summarize_results(results)

    assert summary["total"] == 2.0
    assert summary["answer_accuracy"] == 0.5
    assert summary["retrieval_hit_rate"] == 0.5
    assert summary["parsed_answer_rate"] == 1.0


def load_lawqa_samples_from_dict():
    data = {
        "samples": [
            {
                "ファイル名": "sample-1",
                "コンテキスト": "根拠本文",
                "問題文": "正しいものはどれですか。",
                "指示": "選択肢から選んでください。",
                "選択肢": "a 誤り\nb 正しい\nc 誤り\nd 誤り",
                "output": "b",
            }
        ]
    }
    item = data["samples"][0]
    from scripts.eval_lawqa_jp import LawqaSample

    return LawqaSample(
        filename=item["ファイル名"],
        context=item["コンテキスト"],
        question=item["問題文"],
        instruction=item["指示"],
        choices=item["選択肢"],
        answer_label=item["output"],
    )
