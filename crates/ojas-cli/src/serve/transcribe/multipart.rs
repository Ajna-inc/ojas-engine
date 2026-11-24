//! `multipart/form-data` (RFC 7578): the parts of a request body, borrowed from it.

/// One form field or file.
#[derive(Debug, PartialEq)]
pub(crate) struct Part<'a> {
    pub(crate) name: String,
    pub(crate) filename: Option<String>,
    pub(crate) content_type: Option<String>,
    pub(crate) data: &'a [u8],
}

/// A header value's `; key=value` parameters, quoted values unescaped. A `;` or `=`
/// inside quotes belongs to the value.
fn params(value: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut chars = value.chars().peekable();
    // The value before the first `;` is not a parameter.
    for c in chars.by_ref() { if c == ';' { break; } }
    loop {
        let mut key = String::new();
        while let Some(&c) = chars.peek() {
            if c == '=' || c == ';' { break; }
            key.push(c);
            chars.next();
        }
        let mut val = String::new();
        if chars.next() == Some('=') {
            if chars.peek() == Some(&'"') {
                chars.next();
                while let Some(c) = chars.next() {
                    match c {
                        '"' => break,
                        // Only a quote or a backslash is escaped; Windows paths keep theirs.
                        '\\' if matches!(chars.peek(), Some('"' | '\\')) => val.extend(chars.next()),
                        c => val.push(c),
                    }
                }
                for c in chars.by_ref() { if c == ';' { break; } }
            } else {
                for c in chars.by_ref() {
                    if c == ';' { break; }
                    val.push(c);
                }
                val = val.trim().to_string();
            }
        }
        let key = key.trim().to_ascii_lowercase();
        if !key.is_empty() { out.push((key, val)); }
        if chars.peek().is_none() { return out; }
    }
}

/// The boundary of a `multipart/form-data` content type; `None` for any other type.
pub(crate) fn boundary(content_type: &str) -> Option<String> {
    let kind = content_type.split(';').next().unwrap_or("").trim();
    if !kind.eq_ignore_ascii_case("multipart/form-data") { return None; }
    params(content_type).into_iter().find(|(k, v)| k == "boundary" && !v.is_empty()).map(|(_, v)| v)
}

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || from > hay.len() { return None; }
    hay[from..].windows(needle.len()).position(|w| w == needle).map(|i| i + from)
}

/// The parts of `body`, in order. Content is taken byte for byte, up to the line
/// break before the next boundary.
pub(crate) fn parse<'a>(body: &'a [u8], boundary: &str) -> Result<Vec<Part<'a>>, String> {
    let open = format!("--{boundary}").into_bytes();
    let next = format!("\r\n--{boundary}").into_bytes();
    let mut at = find(body, &open, 0).filter(|i| *i == 0 || body[..*i].ends_with(b"\r\n"))
        .ok_or("the multipart body does not start with its boundary")? + open.len();
    let mut parts = Vec::new();
    loop {
        if body[at..].starts_with(b"--") { return Ok(parts); }
        // Padding may follow a boundary on its line.
        let line_end = find(body, b"\r\n", at).ok_or("the multipart body ends inside a boundary line")?;
        if !body[at..line_end].iter().all(|b| matches!(b, b' ' | b'\t')) {
            return Err("a multipart boundary line carries unexpected bytes".into());
        }
        // Headers end at the first blank line; a part may have none.
        let (head, start) = if body[line_end..].starts_with(b"\r\n\r\n") { (&body[..0], line_end + 4) } else {
            let end = find(body, b"\r\n\r\n", line_end).ok_or("a multipart part has no end to its headers")?;
            (&body[line_end + 2..end], end + 4)
        };
        let close = find(body, &next, start - 2).ok_or("the multipart body has no closing boundary")?;
        let (mut name, mut filename, mut content_type) = (None, None, None);
        for line in String::from_utf8_lossy(head).split("\r\n") {
            let Some((key, value)) = line.split_once(':') else { continue };
            if key.trim().eq_ignore_ascii_case("content-disposition") {
                for (k, v) in params(value) {
                    match k.as_str() {
                        "name" => name = Some(v),
                        "filename" => filename = Some(v),
                        _ => {}
                    }
                }
            } else if key.trim().eq_ignore_ascii_case("content-type") {
                content_type = Some(value.trim().to_string());
            }
        }
        let name = name.ok_or("a multipart part has no name in its Content-Disposition")?;
        parts.push(Part { name, filename, content_type, data: &body[start.min(close)..close] });
        at = close + next.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_boundary_is_read_plain_or_quoted() {
        assert_eq!(boundary("multipart/form-data; boundary=----abc123").as_deref(), Some("----abc123"));
        assert_eq!(boundary("Multipart/Form-Data; charset=utf-8; boundary=\"a b;c\"").as_deref(), Some("a b;c"));
        assert_eq!(boundary("multipart/form-data"), None);
        assert_eq!(boundary("application/json; boundary=x"), None);
    }

    #[test]
    fn fields_and_files_come_out_in_order() {
        let body = b"preamble\r\n--XX\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\nverbose_json\r\n\
                     --XX\r\ncontent-disposition: form-data; name=file; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFFdata\r\n\
                     --XX\r\nContent-Disposition: form-data; name=\"empty\"\r\n\r\n\r\n--XX--\r\nepilogue";
        let parts = parse(body, "XX").unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], Part { name: "response_format".into(), filename: None, content_type: None, data: b"verbose_json" });
        assert_eq!(parts[1], Part { name: "file".into(), filename: Some("a.wav".into()), content_type: Some("audio/wav".into()),
                                    data: b"RIFFdata" });
        assert_eq!((parts[2].name.as_str(), parts[2].data), ("empty", &b""[..]));
    }

    #[test]
    fn quoted_names_keep_their_separators_and_escapes() {
        let body = b"--b\r\nContent-Disposition: form-data; name=\"timestamp_granularities[]\"; filename=\"my; \\\"best\\\" take=1.wav\"\r\n\r\nx\r\n--b--\r\n";
        let parts = parse(body, "b").unwrap();
        assert_eq!(parts[0].name, "timestamp_granularities[]");
        assert_eq!(parts[0].filename.as_deref(), Some("my; \"best\" take=1.wav"));
        let body = b"--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"C:\\audio\\a.wav\"\r\n\r\nx\r\n--b--\r\n";
        assert_eq!(parse(body, "b").unwrap()[0].filename.as_deref(), Some("C:\\audio\\a.wav"));
    }

    #[test]
    fn binary_content_is_kept_byte_for_byte() {
        // Every byte value, line breaks, and text that looks like a boundary but is not one.
        let mut data: Vec<u8> = (0..=255u8).collect();
        data.extend_from_slice(b"\r\n--bound\r\n\r\n--boundar\r\n--");
        let mut body = b"--boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x\"\r\n\r\n".to_vec();
        body.extend_from_slice(&data);
        body.extend_from_slice(b"\r\n--boundary--\r\n");
        let parts = parse(&body, "boundary").unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].data, &data[..]);
    }

    #[test]
    fn broken_bodies_are_refused() {
        assert!(parse(b"no boundary here", "b").is_err());
        assert!(parse(b"--b\r\nContent-Disposition: form-data; name=\"a\"\r\n\r\nnever closed", "b").is_err());
        assert!(parse(b"--b\r\nContent-Type: text/plain\r\n\r\nx\r\n--b--", "b").is_err(), "a part needs a name");
        assert!(parse(b"--b", "b").is_err());
        assert_eq!(parse(b"--b--", "b").unwrap(), Vec::new());
    }
}
