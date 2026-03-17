use std::borrow::Cow;

/// Split a string into shell-like tokens, respecting single and double quotes.
pub(crate) fn split(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut chars = input.chars().peekable();
    let mut in_single = false;
    let mut in_double = false;

    while let Some(c) = chars.next() {
        match c {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            ' ' | '\t' if !in_single && !in_double => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(c),
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

pub(crate) fn quote(value: &str) -> Cow<'_, str> {
    if !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | '/' | ':'))
    {
        return Cow::Borrowed(value);
    }

    let escaped = value.replace('\'', r"'\''");
    Cow::Owned(format!("'{escaped}'"))
}

pub(crate) fn join<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    values
        .into_iter()
        .map(|value| quote(value).into_owned())
        .collect::<Vec<_>>()
        .join(" ")
}
