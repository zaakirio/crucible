"""Interactive terminal front end for Crucible.

Launched by bare `crucible` / `crucible tui`. A home menu (run an eval, browse runs,
compare two runs, pull a model) wraps the same functions the CLI subcommands use - this
is a different front end, not a different pipeline or a different set of features.

Styling: keyboard-first, no button chrome - option lists and Enter-to-submit inputs,
same as opencode's TUI. The `ansi-dark` theme is used deliberately so colors come from
the user's own terminal palette instead of a hardcoded one.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Markdown,
    OptionList,
    ProgressBar,
    RichLog,
    Static,
)
from textual.widgets.option_list import Option

from . import client, db, doctor, hub, server
from .compare import build_comparison_rows
from .eval import EvalReporter, detect_judge, run_eval

MODELS_DIR = Path("models")
TESTS_DIR = "tests"
DB_PATH = "results.db"

HELP_TEXT = """\
# Getting started

## Option A: point at a server that's already running

Ollama, LM Studio, vLLM, or a `llama-server` you started yourself - you just need its
URL (e.g. `http://localhost:11434/v1` for Ollama). Pick **Run a new eval** → **External
server**.

## Option B: let crucible manage a local GGUF for you

1. Put a `.gguf` file under `models/`, relative to wherever you launched `crucible` from.
2. Crucible needs a `llama-server` binary to spawn it. It looks, in order, at:
   - `$CRUCIBLE_LLAMA_SERVER`, if set
   - `llama.cpp/build/bin/llama-server`, walking up from your current directory
   - `llama-server` on your `$PATH`
3. Pick **Run a new eval** → **Local GGUF** - crucible spawns it and tears it down
   for you when the run finishes.

## Judge (grades refusal responses)

There's no default judge - you always pick one. Set `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, or `DEEPSEEK_API_KEY` before launching, or paste a key directly
when asked.

## Pulling a model

**Pull a model from Hugging Face** downloads `.gguf` files straight into `models/` -
just the repo id (e.g. `LiquidAI/LFM2.5-1.2B-Instruct-GGUF`) and an optional filter
(e.g. `Q4_K_M`) to narrow which files.

## Everything here also works as plain flags

`crucible run`, `crucible eval`, `crucible pull`, `crucible compare`, and friends all
still work exactly as before - see `crucible --help` or the README for scripting/CI use.
"""

APP_CSS = """
ModalScreen {
    align: center middle;
}
#dialog {
    width: 76%;
    max-width: 100;
    border-left: thick $accent;
    padding: 0 2;
}
#dialog > Label {
    margin-bottom: 1;
}
#dialog > .hint {
    margin-top: 1;
}
OptionList {
    border: none;
    background: transparent;
    padding: 0 1;
}
#model-list {
    height: auto;
    max-height: 12;
    margin-bottom: 1;
}
#home-body, #preflight-body {
    padding: 1 2;
}
#run-title, #results-footer, #compare-title {
    padding: 1 2;
}
#run-progress {
    margin: 0 2;
}
#run-log {
    margin: 0 2 1 2;
}
DataTable {
    height: 1fr;
}
"""


class _BackToHome(Exception):
    """Raised when a screen is dismissed with None (cancel) - unwinds to the home menu."""


def _hint(text: str) -> Static:
    return Static(f"[dim]{text}[/dim]", classes="hint")


def _safe_list_models(base_url: str) -> list[str]:
    try:
        return client.list_models(base_url)
    except Exception:
        return []


def _load_runs_overview() -> list[tuple]:
    """(run_row, overview_dict) for every stored run, newest first."""
    conn = db.connect(DB_PATH)
    try:
        return [(r, db.run_overview_row(conn, r)) for r in db.list_runs(conn)]
    finally:
        conn.close()


def _load_comparison(run_a: int, run_b: int) -> list:
    conn = db.connect(DB_PATH)
    try:
        sa = {c["category"]: c for c in db.category_summary(conn, run_a)}
        sb = {c["category"]: c for c in db.category_summary(conn, run_b)}
        return build_comparison_rows(sa, sb)
    finally:
        conn.close()


@dataclass
class RunPlan:
    source: str  # "external" | "managed"
    server_url: str | None
    model_path: Path | None
    model_name: str
    base_model_name: str | None
    judge_name: str
    api_key: str


# ── home ──────────────────────────────────────────────────────────────────────

class HomeScreen(Screen[str]):
    BINDINGS = [Binding("q", "quit_app", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="home-body"):
            yield Static("[bold]crucible[/bold]  ·  local-model evidence, not vibes")
            yield OptionList(
                Option("Run a new eval", id="eval"),
                Option("Browse runs", id="runs"),
                Option("Compare two runs", id="compare"),
                Option("Pull a model from Hugging Face", id="pull"),
                Option("Help / getting started", id="help"),
                None,
                Option("Quit", id="quit"),
            )
        yield Footer()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option_id)

    def action_quit_app(self) -> None:
        self.app.exit()


class MessageScreen(ModalScreen[None]):
    """A message plus a way back to the home menu - used for errors and empty states."""

    BINDINGS = [Binding("escape,enter", "quit_app", "Back")]

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self.message)
            yield _hint("enter / esc  back")
        yield Footer()

    def action_quit_app(self) -> None:
        self.dismiss(None)


class HelpScreen(Screen[None]):
    BINDINGS = [Binding("q", "quit_app", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Markdown(HELP_TEXT)
        yield Footer()

    def action_quit_app(self) -> None:
        self.dismiss(None)


# ── preflight ─────────────────────────────────────────────────────────────────

class PreflightScreen(Screen[None]):
    BINDINGS = [Binding("q", "quit_app", "Back"), Binding("enter", "confirm", "Continue")]

    def __init__(self, checks: list[doctor.DoctorCheck]) -> None:
        super().__init__()
        self.checks = checks

    @property
    def has_failures(self) -> bool:
        return any(not c.ok for c in self.checks)

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="preflight-body"):
            yield Static("[bold]crucible doctor[/bold]")
            for c in self.checks:
                if not c.ok:
                    mark = "[bold red]FAIL[/bold red]"
                elif c.warn:
                    mark = "[yellow]warn[/yellow]"
                else:
                    mark = "[green]ok[/green]  "
                yield Static(f"{mark}  {c.name:17} {c.detail}")
            if self.has_failures:
                yield Static("\n[bold red]Fix the above, then try again.[/bold red]")
        yield Footer()

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        if action == "confirm" and self.has_failures:
            return False
        return True

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_quit_app(self) -> None:
        self.dismiss(None)


# ── model source / picking ───────────────────────────────────────────────────

class SourcePickerScreen(ModalScreen[str]):
    BINDINGS = [Binding("escape", "quit_app", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("How do you want to serve the model?")
            yield OptionList(
                Option("External server (Ollama / vLLM / LM Studio / remote llama-server)", id="external"),
                Option("Local GGUF (crucible spawns its own llama-server)", id="managed"),
            )
        yield Footer()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option_id)

    def action_quit_app(self) -> None:
        self.dismiss(None)


class ServerUrlScreen(ModalScreen[str]):
    BINDINGS = [Binding("escape", "quit_app", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Server URL")
            yield Input(placeholder="http://localhost:11434/v1", id="url")
            yield _hint("enter  continue  ·  esc  cancel")
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if value:
            self.dismiss(value)

    def action_quit_app(self) -> None:
        self.dismiss(None)


class ModelPickerScreen(ModalScreen[str]):
    BINDINGS = [Binding("escape", "quit_app", "Cancel")]

    def __init__(self, models: list[str], *, heading: str = "Pick a model") -> None:
        super().__init__()
        self.models = models
        self.heading = heading

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self.heading)
            if self.models:
                with ListView(id="model-list"):
                    for name in self.models:
                        item = ListItem(Label(name))
                        item.model_name = name
                        yield item
            else:
                yield Label("[dim]server didn't list any models - type one below[/dim]")
            yield Input(placeholder="or type a model name", id="manual")
            yield _hint("enter  continue  ·  esc  cancel")
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.model_name)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.value.strip():
            self.dismiss(event.value.strip())

    def action_quit_app(self) -> None:
        self.dismiss(None)


class LocalModelPickerScreen(ModalScreen[Path]):
    BINDINGS = [Binding("escape", "quit_app", "Cancel")]

    def __init__(self, files: list[Path]) -> None:
        super().__init__()
        self.files = files

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Pick a local GGUF")
            if self.files:
                with ListView(id="model-list"):
                    for p in self.files:
                        size_gb = p.stat().st_size / 1e9
                        item = ListItem(Label(f"{size_gb:6.2f} GB  {p}"))
                        item.model_path = p
                        yield item
            else:
                yield Label(f"[dim]no .gguf files under {MODELS_DIR}/ - see Help on the home menu[/dim]")
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.model_path)

    def action_quit_app(self) -> None:
        self.dismiss(None)


class ConfirmBaseScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "quit_app", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Compare against a base model too? (abliteration delta)")
            yield OptionList(Option("Yes", id="yes"), Option("No", id="no"))
        yield Footer()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option_id == "yes")

    def action_quit_app(self) -> None:
        self.dismiss(None)


class JudgeScreen(ModalScreen[tuple[str, str]]):
    """No default judge - always explicit. Key is optional if the matching env var is set."""

    BINDINGS = [Binding("escape", "quit_app", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Judge (grades refusal responses)")
            yield Input(placeholder="claude / openai / deepseek / a URL", id="judge")
            yield Input(placeholder="API key (leave blank to use its env var)", password=True, id="key")
            yield Static("", id="judge-error")
            yield _hint("enter  continue  ·  esc  cancel")
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        judge = self.query_one("#judge", Input).value.strip()
        key = self.query_one("#key", Input).value.strip() or None
        if not judge:
            return
        try:
            resolved_judge, resolved_key = detect_judge(judge, key)
        except ValueError as e:
            self.query_one("#judge-error", Static).update(f"[red]{e}[/red]")
            return
        self.dismiss((resolved_judge, resolved_key))

    def action_quit_app(self) -> None:
        self.dismiss(None)


class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "quit_app", "Cancel")]

    def __init__(self, plan: RunPlan) -> None:
        super().__init__()
        self.plan = plan

    def compose(self) -> ComposeResult:
        p = self.plan
        source_line = (
            f"external: {p.server_url}" if p.source == "external" else f"managed: {p.model_path}"
        )
        with Vertical(id="dialog"):
            yield Label("Ready to run")
            yield Static(f"model    {p.model_name}")
            if p.base_model_name:
                yield Static(f"base     {p.base_model_name}")
            yield Static(f"source   {source_line}")
            yield Static(f"judge    {p.judge_name}")
            yield OptionList(Option("Run", id="run"), Option("Cancel", id="cancel"))
        yield Footer()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option_id == "run")

    def action_quit_app(self) -> None:
        self.dismiss(None)


# ── running + results ─────────────────────────────────────────────────────────

def _execute(plan: RunPlan, reporter: EvalReporter) -> Path:
    if plan.source == "managed":
        with server.llama_server(plan.model_path) as srv:
            return run_eval(
                server_url=srv.base_url,
                model_name=plan.model_name,
                judge=plan.judge_name,
                api_key=plan.api_key,
                tests_dir=TESTS_DIR,
                reporter=reporter,
            )
    return run_eval(
        server_url=plan.server_url,
        model_name=plan.model_name,
        base_model_name=plan.base_model_name,
        judge=plan.judge_name,
        api_key=plan.api_key,
        tests_dir=TESTS_DIR,
        reporter=reporter,
    )


class TextualEvalReporter(EvalReporter):
    """Drives a RunScreen's widgets from run_eval(), called from a worker thread."""

    def __init__(self, screen: "RunScreen") -> None:
        self._screen = screen

    def header(self, *, out_dir: Path, model_name: str,
               base_model_name: str | None, judge_name: str) -> None:
        text = f"[bold]{model_name}[/bold]"
        if base_model_name:
            text += f"  vs  [dim]{base_model_name}[/dim]"
        text += f"   ->  [cyan]{out_dir}/[/cyan]   judge=[dim]{judge_name}[/dim]"
        self._screen.app.call_from_thread(
            self._screen.query_one("#run-title", Static).update, text
        )

    def phase_start(self, label: str, total: int) -> None:
        def _start() -> None:
            self._screen.query_one("#run-progress", ProgressBar).update(total=total, progress=0)
            self._screen.query_one("#run-log", RichLog).write(f"[bold]{label}[/bold]")

        self._screen.app.call_from_thread(_start)

    def phase_tick(self, detail: str) -> None:
        def _tick() -> None:
            self._screen.query_one("#run-progress", ProgressBar).advance(1)
            self._screen.query_one("#run-log", RichLog).write(f"  {detail}")

        self._screen.app.call_from_thread(_tick)

    def phase_done(self) -> None:
        pass  # the matching ticks already bring the bar to total

    def footer(self, *, card_path: Path, report_path: Path) -> None:
        def _footer() -> None:
            self._screen.query_one("#run-log", RichLog).write(
                f"[bold green]done[/bold green]  {card_path}"
            )

        self._screen.app.call_from_thread(_footer)


class RunScreen(Screen[Path | None]):
    BINDINGS = [Binding("q", "quit_app", "Back")]

    def __init__(self, plan: RunPlan) -> None:
        super().__init__()
        self.plan = plan

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(f"Running eval for [bold]{self.plan.model_name}[/bold] ...", id="run-title")
        yield ProgressBar(id="run-progress", show_eta=False)
        yield RichLog(id="run-log", markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.execute()

    @work(exclusive=True, thread=True)
    def execute(self) -> None:
        reporter = TextualEvalReporter(self)
        try:
            out_dir = _execute(self.plan, reporter)
        except Exception as e:
            self.app.call_from_thread(self._fail, str(e))
            return
        self.app.call_from_thread(self.dismiss, out_dir)

    def _fail(self, message: str) -> None:
        self.query_one("#run-log", RichLog).write(f"[bold red]error:[/bold red] {message}")
        self.query_one("#run-title", Static).update("[bold red]Eval failed[/bold red] - press q to go back")

    def action_quit_app(self) -> None:
        self.dismiss(None)


class ResultsScreen(Screen[None]):
    BINDINGS = [Binding("q", "quit_app", "Back")]

    def __init__(self, out_dir: Path) -> None:
        super().__init__()
        self.out_dir = out_dir

    def compose(self) -> ComposeResult:
        card = self.out_dir / "model-card.md"
        text = card.read_text() if card.exists() else "*(no model-card.md found)*"
        yield Header()
        with VerticalScroll():
            yield Markdown(text)
        yield Static(f"report: {self.out_dir / 'report.md'}   (press q to go back)", id="results-footer")
        yield Footer()

    def action_quit_app(self) -> None:
        self.dismiss(None)


# ── browse runs ───────────────────────────────────────────────────────────────

class RunsScreen(Screen[None]):
    BINDINGS = [Binding("q", "quit_app", "Back")]

    def __init__(self, overviews: list[tuple]) -> None:
        super().__init__()
        self.overviews = overviews

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="runs-table")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("id", "status", "model", "lineage", "hardware", "results", "rate", "refusal")
        if not self.overviews:
            table.add_row("-", "-", "no runs yet - run an eval first", "-", "-", "-", "-", "-")
            return
        for run_row, o in self.overviews:
            rate = f"{100 * o['n_passed'] / o['n_graded']:.0f}%" if o["n_graded"] else "-"
            n_labels = o["n_complied"] + o["n_hedged"] + o["n_refused"]
            refusal = f"{o['n_complied']}c/{o['n_hedged']}h/{o['n_refused']}r" if n_labels else ""
            table.add_row(
                str(run_row["id"]), o["status"], f"{run_row['model_name']}[{run_row['quant']}]",
                run_row["lineage"], run_row["hardware"], str(o["n_results"]), rate, refusal,
            )

    def action_quit_app(self) -> None:
        self.dismiss(None)


# ── compare ───────────────────────────────────────────────────────────────────

class RunPickerScreen(ModalScreen[int]):
    BINDINGS = [Binding("escape", "quit_app", "Cancel")]

    def __init__(self, runs: list[tuple], *, heading: str) -> None:
        super().__init__()
        self.runs = runs
        self.heading = heading

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self.heading)
            with ListView(id="model-list"):
                for run_row, _o in self.runs:
                    text = f"#{run_row['id']}  {run_row['model_name']}[{run_row['quant']}]  ({run_row['lineage']})"
                    item = ListItem(Label(text))
                    item.run_id = run_row["id"]
                    yield item
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.run_id)

    def action_quit_app(self) -> None:
        self.dismiss(None)


class CompareResultScreen(Screen[None]):
    BINDINGS = [Binding("q", "quit_app", "Back")]

    def __init__(self, run_a: int, run_b: int, rows: list) -> None:
        super().__init__()
        self.run_a = run_a
        self.run_b = run_b
        self.rows = rows

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(f"compare  A=#{self.run_a}   B=#{self.run_b}", id="compare-title")
        yield DataTable(id="compare-table")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("category", "A", "B", "delta")
        for row in self.rows:
            delta = Text(row.delta, style="bold underline") if row.flagged else Text(row.delta)
            table.add_row(row.category, row.value_a, row.value_b, delta)

    def action_quit_app(self) -> None:
        self.dismiss(None)


# ── pull ──────────────────────────────────────────────────────────────────────

class PullRepoScreen(ModalScreen[tuple[str, str | None]]):
    BINDINGS = [Binding("escape", "quit_app", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Hugging Face repo id")
            yield Input(placeholder="LiquidAI/LFM2.5-1.2B-Instruct-GGUF", id="repo")
            yield Input(placeholder="optional filter, e.g. Q4_K_M", id="filter")
            yield _hint("enter  list files  ·  esc  cancel")
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        repo = self.query_one("#repo", Input).value.strip()
        filt = self.query_one("#filter", Input).value.strip() or None
        if repo:
            self.dismiss((repo, filt))

    def action_quit_app(self) -> None:
        self.dismiss(None)


class PullFileListScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "quit_app", "Cancel"), Binding("enter", "confirm", "Download")]

    def __init__(self, repo_id: str, files: list) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.files = files

    def compose(self) -> ComposeResult:
        total_gb = sum(f.size for f in self.files) / 1e9
        with Vertical(id="dialog"):
            yield Label(f"{len(self.files)} file(s), {total_gb:.2f} GB total, from {self.repo_id}")
            with VerticalScroll(id="model-list"):
                for f in self.files:
                    yield Static(f"{f.size / 1e9:6.2f} GB  {f.path}")
            yield _hint(f"enter  download all {len(self.files)}  ·  esc  cancel")
        yield Footer()

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_quit_app(self) -> None:
        self.dismiss(None)


class PullProgressScreen(Screen[None]):
    BINDINGS = [Binding("q", "quit_app", "Back")]

    def __init__(self, repo_id: str, files: list) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.files = files

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(f"Pulling from [bold]{self.repo_id}[/bold] ...", id="run-title")
        yield ProgressBar(id="run-progress", show_eta=False)
        yield RichLog(id="run-log", markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.execute()

    @work(exclusive=True, thread=True)
    def execute(self) -> None:
        log = self.query_one("#run-log", RichLog)
        bar = self.query_one("#run-progress", ProgressBar)
        title = self.query_one("#run-title", Static)
        try:
            for f in self.files:
                self.app.call_from_thread(log.write, f"[bold]{f.path}[/bold]  ({f.size / 1e9:.2f} GB)")

                def _on_progress(done: int, total: int) -> None:
                    if total:
                        self.app.call_from_thread(bar.update, total=total, progress=done)

                dest = hub.download(self.repo_id, f, MODELS_DIR, on_progress=_on_progress)
                self.app.call_from_thread(log.write, f"[bold green]done[/bold green]  {dest}")
        except Exception as e:
            self.app.call_from_thread(log.write, f"[bold red]error:[/bold red] {e}")
            self.app.call_from_thread(title.update, "[bold red]Pull failed[/bold red] - press q to go back")
            return
        self.app.call_from_thread(title.update, "[bold green]Pull complete[/bold green] - press q to go back")

    def action_quit_app(self) -> None:
        self.dismiss(None)


# ── app ───────────────────────────────────────────────────────────────────────

class CrucibleApp(App):
    TITLE = "crucible"
    CSS = APP_CSS

    def on_mount(self) -> None:
        self.theme = "ansi-dark"
        self.run_wizard()

    async def _wait_or_home(self, screen):
        result = await self.push_screen_wait(screen)
        if result is None:
            raise _BackToHome
        return result

    @work(exclusive=True)
    async def run_wizard(self) -> None:
        while True:
            action = await self.push_screen_wait(HomeScreen())
            if action == "quit":
                break
            try:
                if action == "eval":
                    await self._run_eval_flow()
                elif action == "runs":
                    await self._run_browse_runs_flow()
                elif action == "compare":
                    await self._run_compare_flow()
                elif action == "pull":
                    await self._run_pull_flow()
                elif action == "help":
                    await self.push_screen_wait(HelpScreen())
            except _BackToHome:
                pass
        self.exit()

    async def _run_eval_flow(self) -> None:
        checks = await asyncio.to_thread(doctor.run_doctor, db_path=DB_PATH, tests_dir=TESTS_DIR)
        await self._wait_or_home(PreflightScreen(checks))

        source = await self._wait_or_home(SourcePickerScreen())

        server_url: str | None = None
        model_path: Path | None = None
        base_model_name: str | None = None

        if source == "external":
            server_url = await self._wait_or_home(ServerUrlScreen())
            models = await asyncio.to_thread(_safe_list_models, server_url)
            model_name = await self._wait_or_home(ModelPickerScreen(models))
            want_base = await self._wait_or_home(ConfirmBaseScreen())
            if want_base:
                base_model_name = await self._wait_or_home(
                    ModelPickerScreen(models, heading="Pick a base model to compare against")
                )
        else:
            files = await asyncio.to_thread(server.list_gguf_files, MODELS_DIR)
            model_path = await self._wait_or_home(LocalModelPickerScreen(files))
            model_name = model_path.stem

        judge_name, api_key = await self._wait_or_home(JudgeScreen())

        plan = RunPlan(
            source=source,
            server_url=server_url,
            model_path=model_path,
            model_name=model_name,
            base_model_name=base_model_name,
            judge_name=judge_name,
            api_key=api_key,
        )

        proceed = await self._wait_or_home(ConfirmScreen(plan))
        if not proceed:
            return

        out_dir = await self._wait_or_home(RunScreen(plan))
        await self._wait_or_home(ResultsScreen(out_dir))

    async def _run_browse_runs_flow(self) -> None:
        overviews = await asyncio.to_thread(_load_runs_overview)
        await self._wait_or_home(RunsScreen(overviews))

    async def _run_compare_flow(self) -> None:
        overviews = await asyncio.to_thread(_load_runs_overview)
        finished = [(r, o) for r, o in overviews if r["finished_at"]]
        if len(finished) < 2:
            await self._wait_or_home(
                MessageScreen("Need at least two finished runs to compare - run an eval first.")
            )
            return
        run_a = await self._wait_or_home(RunPickerScreen(finished, heading="Pick run A (baseline)"))
        run_b = await self._wait_or_home(RunPickerScreen(finished, heading="Pick run B (candidate)"))
        rows = await asyncio.to_thread(_load_comparison, run_a, run_b)
        await self._wait_or_home(CompareResultScreen(run_a, run_b, rows))

    async def _run_pull_flow(self) -> None:
        repo_id, filter_str = await self._wait_or_home(PullRepoScreen())
        try:
            files = await asyncio.to_thread(hub.list_ggufs, repo_id)
        except RuntimeError as e:
            await self._wait_or_home(MessageScreen(str(e)))
            return
        if filter_str:
            files = [f for f in files if filter_str.lower() in f.path.lower()]
        if not files:
            await self._wait_or_home(
                MessageScreen(f"No .gguf files matching {filter_str!r} in {repo_id}")
            )
            return
        proceed = await self._wait_or_home(PullFileListScreen(repo_id, files))
        if not proceed:
            return
        await self._wait_or_home(PullProgressScreen(repo_id, files))


def run_app() -> int:
    CrucibleApp().run()
    return 0
