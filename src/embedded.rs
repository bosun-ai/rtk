use anyhow::{Result, anyhow, bail};
use std::{str::FromStr, sync::OnceLock};

use crate::{
    cargo_cmd, diff_cmd,
    discover::registry,
    filter::FilterLevel,
    git, grep_cmd, ls, prettier_cmd, pytest_cmd, read, ruff_cmd, shell_words, toml_filter, tree,
    tsc_cmd,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedStep {
    shell: String,
}

impl PlannedStep {
    fn new(shell: String) -> Self {
        Self { shell }
    }

    #[must_use]
    pub fn shell_command(&self) -> &str {
        &self.shell
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StepResult<'a> {
    pub output: &'a str,
    pub succeeded: bool,
}

#[derive(Debug, Clone)]
pub struct ShellPlan {
    steps: Vec<PlannedStep>,
    renderer: Renderer,
}

impl ShellPlan {
    fn new(steps: Vec<PlannedStep>, renderer: Renderer) -> Self {
        Self { steps, renderer }
    }

    #[must_use]
    pub fn steps(&self) -> &[PlannedStep] {
        &self.steps
    }

    pub fn render(&self, results: &[StepResult<'_>]) -> Result<String> {
        if results.len() != self.steps.len() {
            bail!(
                "expected {} RTK step results, got {}",
                self.steps.len(),
                results.len()
            );
        }

        match &self.renderer {
            Renderer::Cargo(renderer) => Ok(match renderer {
                CargoRenderer::Build => cargo_cmd::filter_cargo_build(results[0].output),
                CargoRenderer::Test => cargo_cmd::filter_cargo_test(results[0].output),
                CargoRenderer::Clippy => cargo_cmd::filter_cargo_clippy(results[0].output),
                CargoRenderer::Install => cargo_cmd::filter_cargo_install(results[0].output),
                CargoRenderer::Passthrough => results[0].output.trim().to_string(),
            }),
            Renderer::Diff => Ok(render_diff_output(results[0].output)),
            Renderer::Git(renderer) => render_git_output(renderer, results),
            Renderer::Grep {
                pattern,
                path,
                max_line_len,
                max_results,
                context_only,
            } => Ok(grep_cmd::render_grep_output(
                results[0].output,
                pattern,
                path,
                *max_line_len,
                *max_results,
                *context_only,
                results[0].succeeded,
            )),
            Renderer::Ls { show_all } => Ok(ls::render_ls_output(results[0].output, *show_all)),
            Renderer::Prettier => Ok(prettier_cmd::filter_prettier_output(results[0].output)),
            Renderer::Pytest => Ok(pytest_cmd::filter_pytest_output(results[0].output)),
            Renderer::Read {
                file_name_hint,
                level,
                max_lines,
                tail_lines,
                line_numbers,
            } => Ok(read::render_read_output(
                results[0].output,
                file_name_hint.as_deref(),
                *level,
                *max_lines,
                *tail_lines,
                *line_numbers,
            )),
            Renderer::Ruff(renderer) => Ok(match renderer {
                RuffRenderer::Check => ruff_cmd::filter_ruff_check_json(results[0].output),
                RuffRenderer::Format => ruff_cmd::filter_ruff_format(results[0].output),
                RuffRenderer::Passthrough => results[0].output.trim().to_string(),
            }),
            Renderer::TomlFilter { command } => {
                let filter = toml_filter::find_filter_in(command, &builtin_toml_registry().filters)
                    .ok_or_else(|| anyhow!("missing builtin TOML filter for command: {command}"))?;

                Ok(toml_filter::apply_filter(filter, results[0].output))
            }
            Renderer::Tree => Ok(tree::filter_tree_output(results[0].output)),
            Renderer::Tsc => Ok(tsc_cmd::filter_tsc_output(results[0].output)),
        }
    }
}

#[derive(Debug, Clone)]
enum Renderer {
    Cargo(CargoRenderer),
    Diff,
    Git(GitRenderer),
    Grep {
        pattern: String,
        path: String,
        max_line_len: usize,
        max_results: usize,
        context_only: bool,
    },
    Ls {
        show_all: bool,
    },
    Prettier,
    Pytest,
    Read {
        file_name_hint: Option<String>,
        level: FilterLevel,
        max_lines: Option<usize>,
        tail_lines: Option<usize>,
        line_numbers: bool,
    },
    Ruff(RuffRenderer),
    TomlFilter {
        command: String,
    },
    Tree,
    Tsc,
}

#[derive(Debug, Clone)]
enum CargoRenderer {
    Build,
    Test,
    Clippy,
    Install,
    Passthrough,
}

#[derive(Debug, Clone)]
enum RuffRenderer {
    Check,
    Format,
    Passthrough,
}

#[derive(Debug, Clone)]
enum GitRenderer {
    Add,
    Branch { write: bool },
    Commit,
    Diff { max_lines: usize },
    Fetch,
    Log { limit: usize, user_set_limit: bool },
    Pull,
    Push,
    Show { max_lines: usize, passthrough: bool },
    Stash(GitStashRenderer),
    Status { compact: bool },
    Worktree { write: bool },
}

#[derive(Debug, Clone)]
enum GitStashRenderer {
    Default,
    Action { subcommand: String },
    List,
    Show,
}

#[must_use]
pub fn plan(raw_command: &str, excluded_commands: &[String]) -> Option<ShellPlan> {
    let rewritten = registry::rewrite_command(raw_command, excluded_commands)?;
    let rewritten_tokens = shell_words::split(&rewritten);
    let actual_command = registry::strip_disabled_prefix(raw_command);
    let raw_tokens = shell_words::split(actual_command);

    if rewritten_tokens.first().map(String::as_str) != Some("rtk") {
        return None;
    }

    match rewritten_tokens.get(1).map(String::as_str) {
        Some("cargo") => plan_cargo_from_raw(raw_command, &raw_tokens).ok(),
        Some("diff") => plan_diff_from_raw(raw_command, &raw_tokens).ok(),
        Some("git") => plan_git_from_raw(raw_command, &raw_tokens).ok(),
        Some("grep") => plan_grep_from_raw(&raw_tokens).ok(),
        Some("ls") => plan_ls(raw_command, raw_tokens.get(1..).unwrap_or_default()).ok(),
        Some("prettier") => Some(plan_passthrough(raw_command, Renderer::Prettier)),
        Some("pytest") => Some(plan_passthrough(raw_command, Renderer::Pytest)),
        Some("read") => plan_read_from_rewritten(raw_command, &rewritten_tokens).ok(),
        Some("ruff") => plan_ruff_from_raw(raw_command, &raw_tokens).ok(),
        Some("tree") => Some(plan_passthrough(raw_command, Renderer::Tree)),
        Some("tsc") => Some(plan_passthrough(raw_command, Renderer::Tsc)),
        _ => plan_toml_filter(raw_command, actual_command),
    }
}

fn plan_passthrough(raw_command: &str, renderer: Renderer) -> ShellPlan {
    ShellPlan::new(
        vec![PlannedStep::new(raw_command.trim().to_string())],
        renderer,
    )
}

fn plan_ls(raw_command: &str, args: &[String]) -> Result<ShellPlan> {
    let show_all = args.iter().any(|arg| {
        (arg.starts_with('-') && !arg.starts_with("--") && arg.contains('a')) || arg == "--all"
    });

    let command = with_env_prefix(raw_command, &ls::build_ls_command(args));

    Ok(ShellPlan::new(
        vec![PlannedStep::new(command)],
        Renderer::Ls { show_all },
    ))
}

fn plan_cargo_from_raw(raw_command: &str, raw_tokens: &[String]) -> Result<ShellPlan> {
    let Some((command, tail)) = raw_tokens.split_first() else {
        bail!("missing cargo command");
    };
    if command != "cargo" {
        bail!("expected raw cargo command");
    }

    let Some((subcommand, _remaining)) = tail.split_first() else {
        bail!("missing cargo subcommand");
    };

    let renderer = match subcommand.as_str() {
        "build" | "check" => CargoRenderer::Build,
        "test" => CargoRenderer::Test,
        "clippy" => CargoRenderer::Clippy,
        "install" => CargoRenderer::Install,
        "fmt" => CargoRenderer::Passthrough,
        _ => return Err(anyhow!("unsupported embedded cargo subcommand: {subcommand}")),
    };

    Ok(plan_passthrough(raw_command, Renderer::Cargo(renderer)))
}

fn plan_read_from_rewritten(raw_command: &str, rewritten_tokens: &[String]) -> Result<ShellPlan> {
    let Some(file_name) = rewritten_tokens.get(2) else {
        bail!("missing read file argument");
    };

    let mut level = FilterLevel::Minimal;
    let mut max_lines = None;
    let mut tail_lines = None;
    let mut line_numbers = false;

    let mut index = 3;
    while index < rewritten_tokens.len() {
        match rewritten_tokens[index].as_str() {
            "--level" | "-l" => {
                let Some(value) = rewritten_tokens.get(index + 1) else {
                    bail!("missing read level value");
                };
                level = FilterLevel::from_str(value).map_err(anyhow::Error::msg)?;
                index += 2;
            }
            "--max-lines" | "-m" => {
                let Some(value) = rewritten_tokens.get(index + 1) else {
                    bail!("missing max-lines value");
                };
                max_lines = Some(value.parse()?);
                index += 2;
            }
            "--tail-lines" => {
                let Some(value) = rewritten_tokens.get(index + 1) else {
                    bail!("missing tail-lines value");
                };
                tail_lines = Some(value.parse()?);
                index += 2;
            }
            "--line-numbers" | "-n" => {
                line_numbers = true;
                index += 1;
            }
            _ => index += 1,
        }
    }

    Ok(plan_passthrough(
        raw_command,
        Renderer::Read {
            file_name_hint: Some(file_name.clone()),
            level,
            max_lines,
            tail_lines,
            line_numbers,
        },
    ))
}

fn plan_diff_from_raw(raw_command: &str, raw_tokens: &[String]) -> Result<ShellPlan> {
    if raw_tokens.len() != 3 || raw_tokens[0] != "diff" {
        bail!("unsupported embedded diff command");
    }

    let command = with_env_prefix(
        raw_command,
        &format!("diff -u {} {}", raw_tokens[1], raw_tokens[2]),
    );

    Ok(ShellPlan::new(
        vec![PlannedStep::new(command)],
        Renderer::Diff,
    ))
}

fn plan_git_from_raw(raw_command: &str, raw_tokens: &[String]) -> Result<ShellPlan> {
    let Some((command, tail)) = raw_tokens.split_first() else {
        bail!("missing git command");
    };
    if command != "git" {
        bail!("expected raw git command");
    }

    let Some((subcommand, remaining)) = tail.split_first() else {
        bail!("missing git subcommand");
    };

    match subcommand.as_str() {
        "add" => {
            let add_command = if remaining.is_empty() {
                with_env_prefix(raw_command, "git add .")
            } else {
                raw_command.trim().to_string()
            };
            Ok(ShellPlan::new(
                vec![
                    PlannedStep::new(add_command),
                    PlannedStep::new(with_env_prefix(raw_command, "git diff --cached --stat --shortstat")),
                ],
                Renderer::Git(GitRenderer::Add),
            ))
        }
        "branch" => {
            let has_action_flag = remaining
                .iter()
                .any(|arg| matches!(arg.as_str(), "-d" | "-D" | "-m" | "-M" | "-c" | "-C"));
            let has_list_flag = remaining.iter().any(|arg| {
                matches!(
                    arg.as_str(),
                    "-a"
                        | "--all"
                        | "-r"
                        | "--remotes"
                        | "--list"
                        | "--merged"
                        | "--no-merged"
                        | "--contains"
                        | "--no-contains"
                )
            });
            let has_positional_arg = remaining.iter().any(|arg| !arg.starts_with('-'));
            let write = has_action_flag || (has_positional_arg && !has_list_flag);

            let command = if write {
                raw_command.trim().to_string()
            } else {
                let list_args = if has_list_flag {
                    format!("git branch --no-color {}", remaining.join(" "))
                } else {
                    format!("git branch -a --no-color {}", remaining.join(" "))
                };
                with_env_prefix(raw_command, list_args.trim())
            };

            Ok(ShellPlan::new(
                vec![PlannedStep::new(command)],
                Renderer::Git(GitRenderer::Branch { write }),
            ))
        }
        "commit" => Ok(plan_passthrough(
            raw_command,
            Renderer::Git(GitRenderer::Commit),
        )),
        "diff" => {
            let wants_stat = remaining
                .iter()
                .any(|arg| matches!(arg.as_str(), "--stat" | "--numstat" | "--shortstat"));
            let wants_compact = !remaining.iter().any(|arg| arg == "--no-compact");

            if wants_stat || !wants_compact {
                return Ok(plan_passthrough(
                    raw_command,
                    Renderer::Git(GitRenderer::Show {
                        max_lines: 500,
                        passthrough: true,
                    }),
                ));
            }

            let suffix = if remaining.is_empty() {
                String::new()
            } else {
                format!(" {}", remaining.join(" "))
            };
            Ok(ShellPlan::new(
                vec![
                    PlannedStep::new(with_env_prefix(raw_command, &format!("git diff --stat{suffix}"))),
                    PlannedStep::new(raw_command.trim().to_string()),
                ],
                Renderer::Git(GitRenderer::Diff { max_lines: 500 }),
            ))
        }
        "fetch" => Ok(plan_passthrough(raw_command, Renderer::Git(GitRenderer::Fetch))),
        "log" => {
            let (command, limit, user_set_limit) = git::build_git_log_command(remaining);
            Ok(ShellPlan::new(
                vec![PlannedStep::new(command)],
                Renderer::Git(GitRenderer::Log {
                    limit,
                    user_set_limit,
                }),
            ))
        }
        "pull" => Ok(plan_passthrough(raw_command, Renderer::Git(GitRenderer::Pull))),
        "push" => Ok(plan_passthrough(raw_command, Renderer::Git(GitRenderer::Push))),
        "show" => {
            let wants_stat_only = remaining
                .iter()
                .any(|arg| matches!(arg.as_str(), "--stat" | "--numstat" | "--shortstat"));
            let wants_format = remaining
                .iter()
                .any(|arg| arg.starts_with("--pretty") || arg.starts_with("--format"));
            let wants_blob_show = remaining
                .iter()
                .any(|arg| !arg.starts_with('-') && arg.contains(':'));

            if wants_stat_only || wants_format || wants_blob_show {
                return Ok(plan_passthrough(
                    raw_command,
                    Renderer::Git(GitRenderer::Show {
                        max_lines: 500,
                        passthrough: true,
                    }),
                ));
            }

            let suffix = if remaining.is_empty() {
                String::new()
            } else {
                format!(" {}", remaining.join(" "))
            };

            Ok(ShellPlan::new(
                vec![
                    PlannedStep::new(with_env_prefix(
                        raw_command,
                        &format!("git show --no-patch --pretty=format:%h\\ %s\\ (%ar)\\ <%an>{suffix}"),
                    )),
                    PlannedStep::new(with_env_prefix(
                        raw_command,
                        &format!("git show --stat --pretty=format:{suffix}"),
                    )),
                    PlannedStep::new(with_env_prefix(
                        raw_command,
                        &format!("git show --pretty=format:{suffix}"),
                    )),
                ],
                Renderer::Git(GitRenderer::Show {
                    max_lines: 500,
                    passthrough: false,
                }),
            ))
        }
        "stash" => {
            let subcommand = remaining.first().map(String::as_str);
            match subcommand {
                Some("list") => Ok(ShellPlan::new(
                    vec![PlannedStep::new(with_env_prefix(raw_command, "git stash list"))],
                    Renderer::Git(GitRenderer::Stash(GitStashRenderer::List)),
                )),
                Some("show") => Ok(ShellPlan::new(
                    vec![PlannedStep::new(with_env_prefix(
                        raw_command,
                        &format!("git stash show -p {}", remaining.join(" ")),
                    ))],
                    Renderer::Git(GitRenderer::Stash(GitStashRenderer::Show)),
                )),
                Some("pop" | "apply" | "drop" | "push") => Ok(plan_passthrough(
                    raw_command,
                    Renderer::Git(GitRenderer::Stash(GitStashRenderer::Action {
                        subcommand: subcommand.expect("subcommand exists").to_string(),
                    })),
                )),
                _ => Ok(plan_passthrough(
                    raw_command,
                    Renderer::Git(GitRenderer::Stash(GitStashRenderer::Default)),
                )),
            }
        }
        "status" => {
            let compact = remaining.is_empty();
            let command = if compact {
                with_env_prefix(raw_command, "git status --porcelain -b")
            } else {
                raw_command.trim().to_string()
            };

            Ok(ShellPlan::new(
                vec![PlannedStep::new(command)],
                Renderer::Git(GitRenderer::Status { compact }),
            ))
        }
        "worktree" => {
            let has_action = remaining
                .iter()
                .any(|arg| matches!(arg.as_str(), "add" | "remove" | "prune" | "lock" | "unlock" | "move"));
            let command = if has_action {
                raw_command.trim().to_string()
            } else {
                with_env_prefix(raw_command, "git worktree list")
            };

            Ok(ShellPlan::new(
                vec![PlannedStep::new(command)],
                Renderer::Git(GitRenderer::Worktree { write: has_action }),
            ))
        }
        _ => Err(anyhow!("unsupported embedded git subcommand: {subcommand}")),
    }
}

fn plan_grep_from_raw(raw_tokens: &[String]) -> Result<ShellPlan> {
    let Some((command, tail)) = raw_tokens.split_first() else {
        bail!("missing grep command");
    };
    if command != "grep" && command != "rg" {
        bail!("expected raw grep/rg command");
    }

    let mut pattern = None;
    let mut path = ".".to_string();
    let mut extra_args = Vec::new();
    let mut index = 0;

    while index < tail.len() {
        let current = &tail[index];
        if pattern.is_none() && current.starts_with('-') {
            extra_args.push(current.clone());
            index += 1;
            continue;
        }

        if pattern.is_none() {
            pattern = Some(current.clone());
            index += 1;
            continue;
        }

        path = current.clone();
        extra_args.extend(tail.iter().skip(index + 1).cloned());
        break;
    }

    let pattern = pattern.ok_or_else(|| anyhow!("missing grep pattern"))?;
    let command = grep_cmd::build_grep_command(&pattern, &path, None, &extra_args);

    Ok(ShellPlan::new(
        vec![PlannedStep::new(command)],
        Renderer::Grep {
            pattern,
            path,
            max_line_len: 80,
            max_results: 50,
            context_only: false,
        },
    ))
}

fn plan_ruff_from_raw(raw_command: &str, raw_tokens: &[String]) -> Result<ShellPlan> {
    let Some((command, args)) = raw_tokens.split_first() else {
        bail!("missing ruff command");
    };
    if command != "ruff" {
        bail!("expected raw ruff command");
    }

    let is_check = args.is_empty()
        || args[0] == "check"
        || (!args[0].starts_with('-') && args[0] != "format" && args[0] != "version");
    let is_format = args.iter().any(|arg| arg == "format");

    if is_check {
        let mut command = String::from("ruff");
        if !args.iter().any(|arg| arg == "--output-format" || arg.starts_with("--output-format=")) {
            command.push_str(" check --output-format=json");
        } else {
            command.push_str(" check");
        }

        let start_index = if args.first().map(String::as_str) == Some("check") {
            1
        } else {
            0
        };
        for arg in &args[start_index..] {
            command.push(' ');
            command.push_str(arg);
        }
        if args[start_index..]
            .iter()
            .all(|arg| arg.starts_with('-') || arg.contains('='))
        {
            command.push_str(" .");
        }

        return Ok(ShellPlan::new(
            vec![PlannedStep::new(with_env_prefix(raw_command, &command))],
            Renderer::Ruff(RuffRenderer::Check),
        ));
    }

    if is_format {
        return Ok(plan_passthrough(
            raw_command,
            Renderer::Ruff(RuffRenderer::Format),
        ));
    }

    Ok(plan_passthrough(
        raw_command,
        Renderer::Ruff(RuffRenderer::Passthrough),
    ))
}

fn plan_toml_filter(raw_command: &str, actual_command: &str) -> Option<ShellPlan> {
    toml_filter::find_filter_in(actual_command, &builtin_toml_registry().filters).map(|_| {
        ShellPlan::new(
            vec![PlannedStep::new(raw_command.trim().to_string())],
            Renderer::TomlFilter {
                command: actual_command.to_string(),
            },
        )
    })
}

fn render_diff_output(output: &str) -> String {
    if output.trim().is_empty() {
        "✅ Files are identical".to_string()
    } else {
        diff_cmd::condense_unified_diff(output)
    }
}

fn render_git_output(renderer: &GitRenderer, results: &[StepResult<'_>]) -> Result<String> {
    Ok(match renderer {
        GitRenderer::Add => {
            if !results[0].succeeded {
                return Ok(results[0].output.trim().to_string());
            }

            let short = results[1].output.lines().last().unwrap_or("").trim();
            if short.is_empty() {
                "ok (nothing to add)".to_string()
            } else {
                format!("ok ✓ {short}")
            }
        }
        GitRenderer::Branch { write } => {
            if *write {
                if results[0].succeeded {
                    "ok ✓".to_string()
                } else {
                    results[0].output.trim().to_string()
                }
            } else {
                git::filter_branch_output(results[0].output)
            }
        }
        GitRenderer::Commit => render_git_commit_output(results[0]),
        GitRenderer::Diff { max_lines } => render_git_diff_output(results[0].output, results[1].output, *max_lines),
        GitRenderer::Fetch => render_git_fetch_output(results[0]),
        GitRenderer::Log {
            limit,
            user_set_limit,
        } => git::render_git_log_output(results[0].output, *limit, *user_set_limit),
        GitRenderer::Pull => render_git_pull_output(results[0]),
        GitRenderer::Push => render_git_push_output(results[0]),
        GitRenderer::Show {
            max_lines,
            passthrough,
        } => {
            if *passthrough {
                results[0].output.trim().to_string()
            } else {
                render_git_show_output(
                    results[0].output,
                    results[1].output,
                    results[2].output,
                    *max_lines,
                )
            }
        }
        GitRenderer::Stash(renderer) => render_git_stash_output(renderer, results[0]),
        GitRenderer::Status { compact } => {
            if *compact {
                git::format_status_output(results[0].output)
            } else {
                git::filter_status_with_args(results[0].output)
            }
        }
        GitRenderer::Worktree { write } => {
            if *write {
                if results[0].succeeded {
                    "ok ✓".to_string()
                } else {
                    results[0].output.trim().to_string()
                }
            } else {
                git::filter_worktree_list(results[0].output)
            }
        }
    })
}

fn render_git_diff_output(stat_output: &str, diff_output: &str, max_lines: usize) -> String {
    let stat = stat_output.trim();
    let diff = diff_output.trim();
    let compacted = git::compact_diff(diff, max_lines);

    match (stat.is_empty(), compacted.is_empty()) {
        (true, true) => String::new(),
        (false, true) => stat.to_string(),
        (true, false) => compacted,
        (false, false) => format!("{stat}\n\n--- Changes ---\n{compacted}"),
    }
}

fn render_git_show_output(summary_output: &str, stat_output: &str, diff_output: &str, max_lines: usize) -> String {
    let mut parts = Vec::new();
    let summary = summary_output.trim();
    let stat = stat_output.trim();
    let diff = diff_output.trim();

    if !summary.is_empty() {
        parts.push(summary.to_string());
    }
    if !stat.is_empty() {
        parts.push(stat.to_string());
    }
    if !diff.is_empty() {
        let compacted = git::compact_diff(diff, max_lines);
        if !compacted.is_empty() {
            parts.push(compacted);
        }
    }

    parts.join("\n")
}

fn render_git_commit_output(result: StepResult<'_>) -> String {
    let output = result.output.trim();
    if result.succeeded {
        if let Some(line) = output.lines().next() {
            if let Some(closing_bracket) = line.find(']') {
                let header = &line[1..closing_bracket];
                if let Some(hash) = header.split_whitespace().last() {
                    if hash.len() >= 7 {
                        return format!("ok ✓ {}", &hash[..7]);
                    }
                }
            }
        }
        "ok ✓".to_string()
    } else if output.contains("nothing to commit") {
        "ok (nothing to commit)".to_string()
    } else {
        output.to_string()
    }
}

fn render_git_push_output(result: StepResult<'_>) -> String {
    let output = result.output.trim();
    if !result.succeeded {
        return output.to_string();
    }
    if output.contains("Everything up-to-date") {
        return "ok (up-to-date)".to_string();
    }

    for line in output.lines() {
        if line.contains("->") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if let Some(last) = parts.last() {
                return format!("ok ✓ {last}");
            }
        }
    }

    "ok ✓".to_string()
}

fn render_git_pull_output(result: StepResult<'_>) -> String {
    let output = result.output.trim();
    if !result.succeeded {
        return output.to_string();
    }
    if output.contains("Already up to date") || output.contains("Already up-to-date") {
        return "ok (up-to-date)".to_string();
    }

    let mut files = 0;
    let mut insertions = 0;
    let mut deletions = 0;
    for line in output.lines() {
        if line.contains("file") && line.contains("changed") {
            for part in line.split(',') {
                let part = part.trim();
                if part.contains("file") {
                    files = part
                        .split_whitespace()
                        .next()
                        .and_then(|n| n.parse().ok())
                        .unwrap_or(0);
                } else if part.contains("insertion") {
                    insertions = part
                        .split_whitespace()
                        .next()
                        .and_then(|n| n.parse().ok())
                        .unwrap_or(0);
                } else if part.contains("deletion") {
                    deletions = part
                        .split_whitespace()
                        .next()
                        .and_then(|n| n.parse().ok())
                        .unwrap_or(0);
                }
            }
        }
    }

    if files > 0 {
        format!("ok ✓ {files} files +{insertions} -{deletions}")
    } else {
        "ok ✓".to_string()
    }
}

fn render_git_fetch_output(result: StepResult<'_>) -> String {
    let output = result.output.trim();
    if !result.succeeded {
        return output.to_string();
    }

    let new_refs = output
        .lines()
        .filter(|line| line.contains("->") || line.contains("[new"))
        .count();

    if new_refs > 0 {
        format!("ok fetched ({new_refs} new refs)")
    } else {
        "ok fetched".to_string()
    }
}

fn render_git_stash_output(renderer: &GitStashRenderer, result: StepResult<'_>) -> String {
    let output = result.output.trim();
    match renderer {
        GitStashRenderer::Action { subcommand } => {
            if result.succeeded {
                format!("ok stash {subcommand}")
            } else {
                output.to_string()
            }
        }
        GitStashRenderer::Default => {
            if result.succeeded {
                if output.contains("No local changes") {
                    "ok (nothing to stash)".to_string()
                } else {
                    "ok stashed".to_string()
                }
            } else {
                output.to_string()
            }
        }
        GitStashRenderer::List => {
            if output.is_empty() {
                "No stashes".to_string()
            } else {
                git::filter_stash_list(output)
            }
        }
        GitStashRenderer::Show => {
            if output.is_empty() {
                "Empty stash".to_string()
            } else {
                git::compact_diff(output, 100)
            }
        }
    }
}

fn builtin_toml_registry() -> &'static toml_filter::TomlFilterRegistry {
    static REGISTRY: OnceLock<toml_filter::TomlFilterRegistry> = OnceLock::new();
    REGISTRY.get_or_init(toml_filter::TomlFilterRegistry::builtin)
}

fn with_env_prefix(raw_command: &str, command: &str) -> String {
    let trimmed = raw_command.trim();
    let actual = registry::strip_disabled_prefix(raw_command);
    let prefix_len = trimmed.len().saturating_sub(actual.len());
    let prefix = trimmed[..prefix_len].trim_end();

    if prefix.is_empty() {
        command.to_string()
    } else {
        format!("{prefix} {command}")
    }
}

#[cfg(test)]
mod tests {
    use super::{StepResult, plan};

    #[test]
    fn plans_ls_via_rewrite_registry() {
        let plan = plan("ls -la", &[]).expect("ls should be planned");
        assert_eq!(plan.steps()[0].shell_command(), "ls -la .");
    }

    #[test]
    fn plans_git_log_via_rewrite_registry() {
        let plan = plan("git log -n 5", &[]).expect("git log should be planned");
        assert!(plan.steps()[0].shell_command().starts_with("git log "));
    }

    #[test]
    fn plans_grep_via_rewrite_registry() {
        let plan = plan("grep -rn needle .", &[]).expect("grep should be planned");
        assert!(
            plan.steps()[0]
                .shell_command()
                .contains("command -v rg >/dev/null 2>&1")
        );
    }

    #[test]
    fn plans_git_status_via_rewrite_registry() {
        let plan = plan("git status", &[]).expect("git status should be planned");
        assert_eq!(plan.steps()[0].shell_command(), "git status --porcelain -b");
    }

    #[test]
    fn plans_git_diff_via_rewrite_registry() {
        let plan = plan("git diff", &[]).expect("git diff should be planned");
        assert_eq!(plan.steps().len(), 2);
        assert!(plan.steps()[0].shell_command().starts_with("git diff --stat"));
        assert_eq!(plan.steps()[1].shell_command(), "git diff");
    }

    #[test]
    fn plans_git_stash_list_via_rewrite_registry() {
        let plan = plan("git stash list", &[]).expect("git stash list should be planned");
        assert_eq!(plan.steps()[0].shell_command(), "git stash list");
    }

    #[test]
    fn plans_every_supported_git_subcommand_via_rewrite_registry() {
        let commands = [
            "git status",
            "git status --short",
            "git log -n 5",
            "git diff",
            "git show HEAD",
            "git add tracked.txt",
            "git commit -m test",
            "git push origin main",
            "git pull --rebase",
            "git branch",
            "git branch feature/test",
            "git fetch --all",
            "git stash",
            "git stash list",
            "git stash show stash@{0}",
            "git stash pop",
            "git worktree list",
            "git worktree add ../feature-worktree",
        ];

        for command in commands {
            assert!(
                plan(command, &[]).is_some(),
                "expected embedded planner to support `{command}`"
            );
        }
    }

    #[test]
    fn plans_read_from_cat_via_rewrite_registry() {
        let plan = plan("cat src/main.rs", &[]).expect("cat should be planned");
        assert_eq!(plan.steps()[0].shell_command(), "cat src/main.rs");
    }

    #[test]
    fn plans_head_from_rewrite_registry() {
        let plan = plan("head -20 src/main.rs", &[]).expect("head should be planned");
        assert_eq!(plan.steps()[0].shell_command(), "head -20 src/main.rs");
    }

    #[test]
    fn plans_diff_via_rewrite_registry() {
        let plan = plan("diff left.txt right.txt", &[]).expect("diff should be planned");
        assert_eq!(plan.steps()[0].shell_command(), "diff -u left.txt right.txt");
    }

    #[test]
    fn plans_ruff_check_with_json_output() {
        let plan = plan("ruff check src", &[]).expect("ruff check should be planned");
        assert_eq!(
            plan.steps()[0].shell_command(),
            "ruff check --output-format=json src"
        );
    }

    #[test]
    fn plans_pytest_via_rewrite_registry() {
        let plan = plan("python -m pytest tests/", &[]).expect("pytest should be planned");
        assert_eq!(plan.steps()[0].shell_command(), "python -m pytest tests/");
    }

    #[test]
    fn plans_tsc_via_rewrite_registry() {
        let plan = plan("npx tsc --noEmit", &[]).expect("tsc should be planned");
        assert_eq!(plan.steps()[0].shell_command(), "npx tsc --noEmit");
    }

    #[test]
    fn plans_prettier_via_rewrite_registry() {
        let plan = plan("npx prettier --check src/", &[]).expect("prettier should be planned");
        assert_eq!(plan.steps()[0].shell_command(), "npx prettier --check src/");
    }

    #[test]
    fn plans_make_via_builtin_toml_filters() {
        let plan = plan("make", &[]).expect("make should be planned");
        assert_eq!(plan.steps()[0].shell_command(), "make");
    }

    #[test]
    fn renders_grep_zero_results_from_empty_output() {
        let plan = plan("grep -rn missing .", &[]).expect("grep should be planned");
        let rendered = plan
            .render(&[StepResult {
                output: "",
                succeeded: false,
            }])
            .expect("render should succeed");

        assert_eq!(rendered, "🔍 0 for 'missing'");
    }

    #[test]
    fn renders_read_output_from_head_rewrite() {
        let plan = plan("head -2 notes.txt", &[]).expect("head should be planned");
        let rendered = plan
            .render(&[StepResult {
                output: "one\ntwo\nthree\n",
                succeeded: true,
            }])
            .expect("render should succeed");

        assert!(rendered.contains("one"));
        assert!(rendered.contains("more lines"));
    }

    #[test]
    fn renders_diff_output_from_unified_diff() {
        let plan = plan("diff left.txt right.txt", &[]).expect("diff should be planned");
        let rendered = plan
            .render(&[StepResult {
                output: "--- left.txt\n+++ right.txt\n@@ -1,2 +1,2 @@\n-old\n+new\n stable\n",
                succeeded: false,
            }])
            .expect("render should succeed");

        assert!(rendered.contains("📄 right.txt (+1 -1)"));
    }

    #[test]
    fn renders_make_with_builtin_toml_filters() {
        let plan = plan("make", &[]).expect("make should be planned");
        let rendered = plan
            .render(&[StepResult {
                output: "make[1]: Entering directory '/tmp'\nbuilt\nmake[1]: Leaving directory '/tmp'\n",
                succeeded: true,
            }])
            .expect("render should succeed");

        assert_eq!(rendered, "built");
    }

    #[test]
    fn renders_git_add_summary_from_follow_up_stat() {
        let plan = plan("git add tracked.txt", &[]).expect("git add should be planned");
        let rendered = plan
            .render(&[
                StepResult {
                    output: "",
                    succeeded: true,
                },
                StepResult {
                    output: " tracked.txt | 2 +-\n 1 file changed, 1 insertion(+), 1 deletion(-)\n",
                    succeeded: true,
                },
            ])
            .expect("render should succeed");

        assert_eq!(rendered, "ok ✓ 1 file changed, 1 insertion(+), 1 deletion(-)");
    }

    #[test]
    fn renders_git_commit_summary_from_commit_output() {
        let plan = plan("git commit -m test", &[]).expect("git commit should be planned");
        let rendered = plan
            .render(&[StepResult {
                output: "[main abc1234] test message\n 1 file changed, 1 insertion(+)\n",
                succeeded: true,
            }])
            .expect("render should succeed");

        assert_eq!(rendered, "ok ✓ abc1234");
    }

    #[test]
    fn renders_git_push_up_to_date_summary() {
        let plan = plan("git push origin main", &[]).expect("git push should be planned");
        let rendered = plan
            .render(&[StepResult {
                output: "Everything up-to-date\n",
                succeeded: true,
            }])
            .expect("render should succeed");

        assert_eq!(rendered, "ok (up-to-date)");
    }

    #[test]
    fn renders_git_pull_up_to_date_summary() {
        let plan = plan("git pull --rebase", &[]).expect("git pull should be planned");
        let rendered = plan
            .render(&[StepResult {
                output: "Already up to date.\n",
                succeeded: true,
            }])
            .expect("render should succeed");

        assert_eq!(rendered, "ok (up-to-date)");
    }

    #[test]
    fn renders_git_fetch_summary_with_new_refs() {
        let plan = plan("git fetch --all", &[]).expect("git fetch should be planned");
        let rendered = plan
            .render(&[StepResult {
                output: " * [new branch]      feature/test -> origin/feature/test\n",
                succeeded: true,
            }])
            .expect("render should succeed");

        assert_eq!(rendered, "ok fetched (1 new refs)");
    }

    #[test]
    fn renders_git_stash_default_summary() {
        let plan = plan("git stash", &[]).expect("git stash should be planned");
        let rendered = plan
            .render(&[StepResult {
                output: "Saved working directory and index state WIP on main: abc1234 test\n",
                succeeded: true,
            }])
            .expect("render should succeed");

        assert_eq!(rendered, "ok stashed");
    }

    #[test]
    fn renders_git_stash_show_as_compact_diff() {
        let plan = plan("git stash show stash@{0}", &[]).expect("git stash show should be planned");
        let rendered = plan
            .render(&[StepResult {
                output: "diff --git a/file.txt b/file.txt\n--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new\n",
                succeeded: true,
            }])
            .expect("render should succeed");

        assert!(rendered.contains("📄 file.txt"));
    }

    #[test]
    fn renders_git_worktree_list_compactly() {
        let plan = plan("git worktree list", &[]).expect("git worktree list should be planned");
        let rendered = plan
            .render(&[StepResult {
                output: "/tmp/repo  abc1234 [main]\n/tmp/repo-feature  def5678 [feature/test]\n",
                succeeded: true,
            }])
            .expect("render should succeed");

        assert!(rendered.contains("/tmp/repo abc1234 [main]"));
        assert!(rendered.contains("/tmp/repo-feature def5678 [feature/test]"));
    }
}
