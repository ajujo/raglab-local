"""Tests for operational commands exposed in the rag-lab CLI (v1.16.2).

Covers:
- rag-lab --help shows doctor, benchmark, reconcile, diagnose
- rag-lab <cmd> --help forwards to argparse and shows correct options
- rag-lab benchmark --suite / --variants / --no-cache accepted (no --variant singular)
- rag-lab benchmark run is NOT a valid sub-command
- python -m entry points still callable via main(argv=[...])
"""

import pytest
from typer.testing import CliRunner

from rag_lab.cli import app


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# rag-lab --help presence checks
# ---------------------------------------------------------------------------

class TestMainHelpShowsOpsCommands:
    def test_doctor_in_help(self, runner):
        result = runner.invoke(app, ["--help"])
        assert "doctor" in result.output

    def test_benchmark_in_help(self, runner):
        result = runner.invoke(app, ["--help"])
        assert "benchmark" in result.output

    def test_reconcile_in_help(self, runner):
        result = runner.invoke(app, ["--help"])
        assert "reconcile" in result.output

    def test_diagnose_in_help(self, runner):
        result = runner.invoke(app, ["--help"])
        assert "diagnose" in result.output


# ---------------------------------------------------------------------------
# rag-lab <cmd> --help shows argparse options
# ---------------------------------------------------------------------------

class TestSubCommandHelp:
    def test_doctor_help(self, runner):
        result = runner.invoke(app, ["doctor", "--help"])
        assert result.exit_code == 0
        assert "--checks" in result.output

    def test_benchmark_help(self, runner):
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "--variants" in result.output
        assert "--suite" in result.output
        assert "--no-cache" in result.output

    def test_reconcile_help(self, runner):
        result = runner.invoke(app, ["reconcile", "--help"])
        assert result.exit_code == 0
        assert "--check" in result.output
        assert "--repair" in result.output

    def test_diagnose_help(self, runner):
        result = runner.invoke(app, ["diagnose", "--help"])
        assert result.exit_code == 0
        assert "--query" in result.output
        assert "--explain" in result.output


# ---------------------------------------------------------------------------
# benchmark flag correctness
# ---------------------------------------------------------------------------

class TestBenchmarkFlags:
    def test_benchmark_help_has_variants_plural(self, runner):
        result = runner.invoke(app, ["benchmark", "--help"])
        assert "--variants" in result.output

    def test_benchmark_help_no_variant_singular(self, runner):
        """The flag is --variants (plural). --variant singular must not appear."""
        result = runner.invoke(app, ["benchmark", "--help"])
        # --variants will match, but we must not find bare "--variant " as a flag name
        lines = result.output.splitlines()
        flag_lines = [l for l in lines if "--variant " in l and "--variants" not in l]
        assert flag_lines == [], f"Found --variant singular: {flag_lines}"

    def test_benchmark_accepts_suite_flag(self, runner):
        result = runner.invoke(app, ["benchmark", "--help"])
        assert "--suite" in result.output

    def test_benchmark_accepts_no_cache_flag(self, runner):
        result = runner.invoke(app, ["benchmark", "--help"])
        assert "--no-cache" in result.output

    def test_benchmark_run_subcommand_not_documented(self, runner):
        """'benchmark run' must not appear as a documented sub-command."""
        result = runner.invoke(app, ["--help"])
        assert "benchmark run" not in result.output
        # Also verify benchmark is listed but as a direct command, not with 'run'
        assert "benchmark" in result.output


# ---------------------------------------------------------------------------
# python -m entry points — main(argv=[...]) callable
# ---------------------------------------------------------------------------

class TestPythonMEntryPoints:
    def test_doctor_main_callable_with_argv(self):
        from rag_lab.doctor import main as doctor_main
        # --checks config skips stores/chroma/fts5 — fast, no model loading
        code = doctor_main(["--checks", "config"])
        assert code in (0, 1, 2)

    def test_reconcile_main_callable_with_argv(self, tmp_path):
        from rag_lab.maintenance.reconcile import main as reconcile_main
        out = tmp_path / "report.json"
        code = reconcile_main(["--check", "--report-json", str(out)])
        assert code in (0, 1)
        assert out.exists()

    def test_diagnose_main_callable_with_argv(self):
        from rag_lab.maintenance.diagnose import main as diagnose_main
        code = diagnose_main([])
        assert code in (0, 1)

    def test_benchmark_main_already_had_argv(self):
        from rag_lab.benchmark.__main__ import main as bench_main
        # --help exits 0 without running benchmarks
        with pytest.raises(SystemExit) as exc_info:
            bench_main(["--help"])
        assert exc_info.value.code == 0
