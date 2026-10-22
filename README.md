# Credit Memo Agent

Credit Memo Agent creates a preliminary credit memo from the latest SEC Form
10-K for a US-listed company. It retrieves company filings and XBRL financial
data, calculates selected financial metrics, and selects relevant filing
excerpts. Unless facts-only mode is enabled, it also drafts a memo with Gemini
and checks its citations and figures.

> This tool produces an automated preliminary draft, not investment advice or a
> credit rating. Verify all figures and statements against the original filing.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for dependency and project management
- A contact name and email address for SEC requests
- A Gemini API key to generate a narrative memo (optional for facts-only mode)

## Install

From the project directory, sync the project dependencies:

```sh
uv sync
```

Set the SEC user agent in your shell. The SEC requires a contact email:

```sh
export SEC_USER_AGENT="Your Name you@example.com"
```

For narrative memos, set a Gemini key. `GOOGLE_API_KEY` is accepted as an
alternative:

```sh
export GEMINI_API_KEY="your-api-key"
```

On Windows PowerShell, set the variables for the current session with:

```powershell
$env:SEC_USER_AGENT = "Your Name you@example.com"
$env:GEMINI_API_KEY = "your-api-key"
```

## Run

Generate a memo for a ticker such as Apple:

```sh
uv run credit-memo-agent AAPL
```

By default, the report is written to `memo_AAPL.md` in the current directory.
Specify a different output path with `--out`:

```sh
uv run credit-memo-agent AAPL --out reports/apple-credit-memo.md
```

To retrieve the filing and financial information without calling Gemini, use
facts-only mode:

```sh
uv run credit-memo-agent AAPL --facts-only
```

Other options:

```text
--max-repairs N   Maximum automated repair attempts for the generated memo
--selftest        Run offline self-tests; does not require SEC or Gemini access
-h, --help        Show all command-line options
```

## Configuration

| Environment variable | Purpose |
| --- | --- |
| `SEC_USER_AGENT` | Required SEC request identity, including a contact email |
| `GEMINI_API_KEY` | Gemini API key for narrative memo drafting |
| `GOOGLE_API_KEY` | Alternative Gemini API key variable |
| `GEMINI_MODEL` | Gemini model ID override |
| `GEMINI_TEMPERATURE` | Optional model temperature |
| `CMA_CACHE_DIR` | SEC response cache directory (defaults to `.sec_cache`) |
| `GEMINI_INPUT_PRICE_PER_M` | Optional input token price per million, for cost estimates |
| `GEMINI_OUTPUT_PRICE_PER_M` | Optional output token price per million, for cost estimates |

## Tests

Run the built-in offline checks:

```sh
uv run credit-memo-agent --selftest
```

These exercise financial calculations, filing extraction and retrieval, memo
verification, and the draft/repair workflow with mock SEC and LLM responses.

## Project layout

```text
agent/
├── __init__.py     # Public package exports
├── __main__.py     # Enables `python -m agent`
├── cli.py          # Command-line interface
├── config.py       # Prompts and application constants
├── filing.py       # HTML parsing, section extraction, and retrieval
├── financials.py   # XBRL facts and financial ratios
├── llm.py          # Gemini integration
├── memo.py         # Memo context and citation/number verification
├── models.py       # Shared data models
├── sec_client.py   # SEC HTTP access and response cache
├── selftest.py     # Offline end-to-end self-test
└── workflow.py     # LangGraph nodes and workflow
```
