"""
Main entry point for running a RAGAS evaluation.

Usage:
    python run_evaluation.py --pdf path/to/doc.pdf --run-name baseline
    python run_evaluation.py --pdf path/to/doc.pdf --run-name v2_improved_prompt
    python run_evaluation.py --compare results/baseline_*.csv results/v2_*.csv

After your first run, open results/ to see the CSV.
After two runs, use --compare to see the delta report.
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_eval(pdf_path: str, run_name: str) -> str:
    """
    Run the full evaluation pipeline and return the results CSV path.

    Prefers the live RAG API when RAG_API_URL is set in .env.
    Falls back to the local rag-chatbot-langchain chain otherwise.
    """
    from src.evaluator import load_qa_pairs, build_ragas_dataset, run_evaluation, save_results
    from src.dataset_builder import validate_qa_pairs, summarize_dataset
    from src.api_client import get_api_url, build_ragas_dataset_from_api

    qa_pairs = load_qa_pairs()
    errors = validate_qa_pairs(qa_pairs)
    if errors:
        logger.error("Eval dataset has errors — fix them before running:")
        for e in errors:
            logger.error(f"  {e}")
        sys.exit(1)

    summarize_dataset(qa_pairs)

    api_url = get_api_url()

    if api_url:
        logger.info(f"Using live RAG API at {api_url}")
        dataset = build_ragas_dataset_from_api(qa_pairs, api_url, pdf_path=pdf_path)
    else:
        logger.info("RAG_API_URL not set — falling back to local rag-chatbot-langchain chain")
        sys.path.insert(0, "../rag-chatbot-langchain")
        try:
            from src.ingestion import ingest_pdf
            from src.chain import build_rag_chain
        except ImportError:
            logger.error(
                "Could not import from rag-chatbot-langchain and RAG_API_URL is not set. "
                "Set RAG_API_URL=http://localhost:8001 or run Project 1 API server first."
            )
            sys.exit(1)

        logger.info(f"Ingesting PDF: {pdf_path}")
        vectorstore = ingest_pdf(pdf_path, persist_directory="./eval_chroma_db")
        chain, retriever = build_rag_chain(vectorstore)
        dataset = build_ragas_dataset(qa_pairs, chain, retriever)

    results_df = run_evaluation(dataset)
    path = save_results(results_df, run_name)

    from src.reporter import print_single_run_summary
    print_single_run_summary(path)

    return path


def compare(path_a: str, path_b: str) -> None:
    from src.reporter import compare_runs
    compare_runs(path_a, path_b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Evaluation Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    eval_parser = subparsers.add_parser("eval", help="Run evaluation on a PDF")
    eval_parser.add_argument("--pdf", required=True, help="Path to the PDF to evaluate")
    eval_parser.add_argument("--run-name", default="baseline", help="Name for this run")

    cmp_parser = subparsers.add_parser("compare", help="Compare two eval runs")
    cmp_parser.add_argument("baseline", help="Path to baseline results CSV")
    cmp_parser.add_argument("improved", help="Path to improved results CSV")

    args = parser.parse_args()

    if args.command == "eval":
        if not os.path.exists(args.pdf):
            logger.error(f"PDF not found: {args.pdf}")
            sys.exit(1)
        run_eval(args.pdf, args.run_name)

    elif args.command == "compare":
        compare(args.baseline, args.improved)

    else:
        parser.print_help()
